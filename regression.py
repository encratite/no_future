from collections import defaultdict
from itertools import chain
from typing import Iterable
from statistics import mean
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import r2_score as get_r2_score

from common import read_ohlc_series, get_rate_of_change, execute_thread_pool, print_table, format_percentage, get_performance_string
from enums import ModelType
from models import get_models
from ohlc import OhlcRecord
from result import RegressionResult
from series import TimeSeries

def analyze_momentum(symbols: list[str], start: pd.Timestamp, split: pd.Timestamp, end: pd.Timestamp) -> None:
	def execute_symbol(symbol: str) -> list[RegressionResult]:
		return analyze_momentum_by_symbol(symbol, start, split, end)

	pool_results: Iterable[list[RegressionResult]] = execute_thread_pool(execute_symbol, symbols)
	all_results: list[RegressionResult] = list(chain.from_iterable(pool_results))
	results_by_symbol: defaultdict[str, defaultdict[ModelType, list[RegressionResult]]] = defaultdict(lambda: defaultdict(list))
	results_by_model: defaultdict[str, list[RegressionResult]] = defaultdict(list)
	results_by_model_parameters: defaultdict[ModelType, defaultdict[str, list[RegressionResult]]] = defaultdict(lambda: defaultdict(list))
	for result in all_results:
		results_by_symbol[result.symbol][result.model_type].append(result)
		results_by_model[result.model_name].append(result)
		for key, value in result.parameters.items():
			name = f"{result.model_name}({key}={value})"
			results_by_model_parameters[result.model_type][name].append(result)

	model_names: dict[ModelType, str] = {}
	models = get_models()
	for model_name, model_type, _model, _parameters in models:
		model_names[model_type] = model_name

	print_model_table("All", results_by_model)
	for model_type, model_results in results_by_model_parameters.items():
		description = model_names[model_type]
		print_model_table(description, model_results)
	print_symbol_table(results_by_symbol, model_names)

def print_symbol_table(results_by_symbol: defaultdict[str, defaultdict[ModelType, list[RegressionResult]]], model_names: dict[ModelType, str]) -> None:
	headers = [
		"Symbol"
	]
	for model_type in ModelType:
		model_name = model_names[model_type]
		headers.append(model_name)
	table = [headers]
	for symbol, model_results in results_by_symbol.items():
		row = [symbol]
		for model_type, results in model_results.items():
			r2_score = mean([x.r2_score_validation for x in results])
			r2_scores_string = format_percentage(r2_score)
			row.append(r2_scores_string)
		table.append(row)
	print_table(table)

def print_model_table(description: str, model_results: defaultdict[str, list[RegressionResult]]) -> None:
	headers = [
		f"Models ({description})",
		"R^2 (IS)",
		"R^2 (OOS)"
	]
	table = [headers]
	for name, results in model_results.items():
		r2_score_training = mean([x.r2_score_training for x in results])
		r2_score_validation = mean([x.r2_score_validation for x in results])
		row = [
			name,
			format_percentage(r2_score_training),
			format_percentage(r2_score_validation)
		]
		table.append(row)
	print_table(table)

def analyze_momentum_by_symbol(symbol: str, start: pd.Timestamp, split: pd.Timestamp, end: pd.Timestamp) -> list[RegressionResult]:
	assert start < split < end
	ohlc_series = read_ohlc_series(symbol)
	training_times = [x for x in ohlc_series if start <= x < split]
	validation_times = [x for x in ohlc_series if split <= x < end]
	training_samples = len(training_times)
	validation_samples = len(validation_times)
	sequential_returns = 20
	feature_count = sequential_returns
	x_training = np.empty((training_samples, feature_count), dtype=np.float64)
	y_training = np.empty(training_samples, dtype=np.float64)
	x_validation = np.empty((validation_samples, feature_count), dtype=np.float64)
	y_validation = np.empty(validation_samples, dtype=np.float64)
	get_sequential_returns(sequential_returns, ohlc_series, training_times, x_training, y_training)
	get_sequential_returns(sequential_returns, ohlc_series, validation_times, x_validation, y_validation)

	warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

	models = get_models()
	results = []
	for model_name, model_type, model, parameters in models:
		model.fit(x_training, y_training)
		training_predictions = model.predict(x_training)
		validation_predictions = model.predict(x_validation)
		r2_score_training = get_r2_score(y_training, training_predictions)
		r2_score_validation = get_r2_score(y_validation, validation_predictions)
		result = RegressionResult(
			symbol,
			model_name,
			model_type,
			parameters,
			r2_score_training,
			r2_score_validation
		)
		results.append(result)
	return results

def get_sequential_returns(sequential_returns: int, ohlc_series: TimeSeries[OhlcRecord], times: list[pd.Timestamp], x: npt.NDArray, y: npt.NDArray) -> None:
	for i, time in enumerate(times):
		future_record = ohlc_series.get(time + pd.Timedelta(days=1), right=True)
		records = [future_record] + ohlc_series.get(time, count=sequential_returns + 1)
		returns = [get_rate_of_change(a.close, b.close) for a, b in zip(records, records[1:])]
		x[i] = returns[1:]
		y[i] = returns[0]