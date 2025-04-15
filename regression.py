import warnings
from collections import defaultdict
from itertools import chain
from statistics import mean
from typing import Iterable, Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import r2_score as get_r2_score
from sklearn.preprocessing import RobustScaler

from common import read_ohlc_series, get_rate_of_change, execute_thread_pool, print_table, format_percentage
from configuration import Configuration
from enums import ModelType
from models import get_models
from ohlc import OhlcRecord
from result import RegressionResult
from series import TimeSeries

REFERENCE_SYMBOLS: Final[list[str]] = [
	"ES",
	"VI",
	"GC",
	"CL",
	# "NG",
	# "E6",
	# "ZN",
	# "ZT"
]

def perform_regression(
	symbols: list[str],
	start: pd.Timestamp,
	split: pd.Timestamp,
	end: pd.Timestamp,
	pca: int | None,
	select_k_best: int | None
) -> None:
	assert not pca or not select_k_best

	reference_ohlc_dict: dict[str, TimeSeries[OhlcRecord]] = {}
	for symbol in REFERENCE_SYMBOLS:
		reference_ohlc_dict[symbol] = read_ohlc_series(symbol)

	def execute_symbol(x: str) -> list[RegressionResult]:
		return symbol_regression(
			x,
			start,
			split,
			end,
			pca,
			select_k_best,
			reference_ohlc_dict
		)

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

	feature_count = all_results[0].feature_count
	model_names: dict[ModelType, str] = {}
	models = get_models(feature_count)
	for model_name, model_type, _model, _parameters in models:
		model_names[model_type] = model_name

	print_model_table("All", results_by_model)
	for model_type, model_results in results_by_model_parameters.items():
		description = model_names[model_type]
		print_model_table(description, model_results, True)
	print_symbol_table(results_by_symbol, model_names)

def print_symbol_table(results_by_symbol: defaultdict[str, defaultdict[ModelType, list[RegressionResult]]], model_names: dict[ModelType, str]) -> None:
	headers = [
		"Symbol (OOS R^2)"
	]
	for model_type in ModelType:
		if model_type in model_names:
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

def print_model_table(description: str, model_results: defaultdict[str, list[RegressionResult]], sort_by_name: bool=False) -> None:
	headers = [
		f"Models ({description})",
		"IS R^2",
		"OOS R^2"
	]
	table = [headers]

	def get_name_key(x: tuple[str, list[RegressionResult]]) -> str:
		key, _results = x
		return key

	if sort_by_name:
		source = sorted(model_results.items(), key=get_name_key)
	else:
		source = model_results.items()
	for name, results in source:
		r2_score_training = mean([x.r2_score_training for x in results])
		r2_score_validation = mean([x.r2_score_validation for x in results])
		row = [
			name,
			format_percentage(r2_score_training),
			format_percentage(r2_score_validation)
		]
		table.append(row)
	print_table(table)

def symbol_regression(
	symbol: str,
	start: pd.Timestamp,
	split: pd.Timestamp,
	end: pd.Timestamp,
	pca: int | None,
	select_k_best: int | None,
	reference_ohlc_dict: dict[str, TimeSeries[OhlcRecord]]
) -> list[RegressionResult]:
	assert start < split < end
	reference_symbols = set(REFERENCE_SYMBOLS)
	symbol_without_suffix = symbol.replace(".F1", "")
	if symbol_without_suffix in reference_symbols:
		reference_symbols.remove(symbol_without_suffix)
	reference_ohlc_series = [reference_ohlc_dict[x] for x in reference_symbols]
	ohlc_series = read_ohlc_series(symbol)
	training_times = [x for x in ohlc_series if start <= x < split]
	if Configuration.ENABLE_PYTORCH_MODELS:
		days_to_remove = len(training_times) % Configuration.MAX_BATCH_SIZE
		if days_to_remove > 0:
			training_times = training_times[days_to_remove:]
			assert len(training_times) % Configuration.MAX_BATCH_SIZE == 0
	validation_times = [x for x in ohlc_series if split <= x < end]
	training_samples = len(training_times)
	validation_samples = len(validation_times)
	sequential_returns = 7
	day_of_week_features = 5
	feature_count = (1 + len(reference_symbols)) * sequential_returns + day_of_week_features
	x_training = np.empty((training_samples, feature_count), dtype=np.float64)
	y_training = np.empty(training_samples, dtype=np.float64)
	x_validation = np.empty((validation_samples, feature_count), dtype=np.float64)
	y_validation = np.empty(validation_samples, dtype=np.float64)

	get_features(
		sequential_returns,
		ohlc_series,
		reference_ohlc_series,
		training_times,
		x_training,
		y_training
	)
	get_features(
		sequential_returns,
		ohlc_series,
		reference_ohlc_series,
		validation_times,
		x_validation,
		y_validation
	)

	if pca is not None:
		transform = PCA(n_components=pca)
		transform.fit(x_training)
		x_training = transform.transform(x_training)
		x_validation = transform.transform(x_validation)
	elif select_k_best is not None:
		transform = SelectKBest(score_func=mutual_info_regression, k=select_k_best)
		transform.fit(x_training, y_training)
		x_training = transform.transform(x_training)
		x_validation = transform.transform(x_validation)

	transformer = RobustScaler()
	transformer.fit(x_training)
	x_training = transformer.transform(x_training)
	x_validation = transformer.transform(x_validation)

	warnings.filterwarnings("ignore", category=UserWarning)
	warnings.filterwarnings("ignore", category=ConvergenceWarning)

	models = get_models(feature_count)
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
			r2_score_validation,
			feature_count
		)
		results.append(result)
		if Configuration.ENABLE_PYTORCH_MODELS:
			print(f"{model_name}: {parameters}, {format_percentage(r2_score_training)}, {format_percentage(r2_score_validation)}")
	return results

def get_features(
	sequential_returns: int,
	ohlc_series: TimeSeries[OhlcRecord],
	reference_ohlc_series: list[TimeSeries[OhlcRecord]],
	times: list[pd.Timestamp],
	x: npt.NDArray,
	y: npt.NDArray
) -> None:
	for i, time in enumerate(times):
		count = sequential_returns + 1
		records = ohlc_series.get(time, count=count)
		tomorrow = ohlc_series.get(time + pd.Timedelta(days=1), right=True)
		today = records[0]
		features = get_sequential_returns_features(sequential_returns, records)
		for reference in reference_ohlc_series:
			records = reference.get(time, count=count)
			reference_features = get_sequential_returns_features(sequential_returns, records)
			features += reference_features
		for day in range(5):
			features.append(1 if time.day_of_week == day else 0)
		label = get_rate_of_change(tomorrow.close, today.close)
		x[i] = features
		y[i] = label

def get_sequential_returns_features(sequential_returns: int, records: list[OhlcRecord]) -> list[float]:
	features = [get_rate_of_change(a.close, b.close) for a, b in zip(records[:sequential_returns], records[1:sequential_returns + 1])]
	return features