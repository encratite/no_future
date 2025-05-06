from enum import Enum
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from colorama import Fore, Style
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
	SelectKBest,
	mutual_info_regression
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as get_r2_score
from sklearn.preprocessing import StandardScaler

from common import (
	read_ohlc_series,
	get_rate_of_change,
	print_table
)
from strategy import Strategy

HIGH_CORRELATION: Final[float] = 0.07
LOW_CORRELATION: Final[float] = 0.01

class FilterMode(Enum):
	NONE: Final[int] = 0
	POSITIVE: Final[int] = 1
	NEGATIVE: Final[int] = 2
	NEW_HIGH_5: Final[int] = 3
	NEW_LOW_5: Final[int] = 4

class OhlcFeature:
	name: str
	values: list[float]

	def __init__(self, name: str, values: list[float]) -> None:
		self.name = name
		self.values = values

def analyze_ohlc_features(
	symbols: list[str],
	start: pd.Timestamp,
	split: pd.Timestamp,
	end: pd.Timestamp,
	filter_mode: FilterMode,
	pca_features: int | None,
	select_k_best: int | None
) -> None:
	assert pca_features is None or select_k_best is None
	headers = [
		"Symbol"
	]
	table = [headers]
	for i, symbol in enumerate(symbols):
		training_features, training_returns = get_ohlc_features(symbol, start, split, filter_mode)
		validation_features, validation_returns = get_ohlc_features(symbol, split, end, filter_mode)
		if len(training_features[0].values) < 2 or len(validation_features[0].values) < 2:
			print(f"Warning: skipping symbol {symbol} due to lack of features after filtering")
			continue
		row = [symbol]
		for feature in training_features:
			pearson = pearsonr(feature.values, training_returns)
			pearson_string = f"{pearson.statistic:.4f}"
			if abs(pearson.statistic) > HIGH_CORRELATION:
				pearson_string = f"{Fore.GREEN}{pearson_string}{Style.RESET_ALL}"
			elif abs(pearson.statistic) < LOW_CORRELATION:
				pearson_string = f"{Fore.RED}{pearson_string}{Style.RESET_ALL}"
			row.append(pearson_string)
			if i == 0:
				headers.append(feature.name)
		model = LinearRegression()
		x_training, y_training = get_features_labels(training_features, training_returns)
		x_validation, y_validation = get_features_labels(validation_features, validation_returns)
		if pca_features is None:
			scaler = StandardScaler()
			scaler.fit(x_training)
			x_training = scaler.transform(x_training)
			x_validation = scaler.transform(x_validation)
		if pca_features is not None:
			pca = PCA(n_components=pca_features)
			pca.fit(x_training)
			x_training = pca.transform(x_training)
			x_validation = pca.transform(x_validation)
		elif select_k_best is not None:
			selector = SelectKBest(score_func=mutual_info_regression, k=select_k_best)
			selector.fit(x_training, y_training)
			x_training = selector.transform(x_training)
			x_validation = selector.transform(x_validation)
		model.fit(x_training, y_training)
		training_predictions = model.predict(x_training)
		validation_predictions = model.predict(x_validation)
		r2_score_training = get_r2_score(y_training, training_predictions)
		r2_score_validation = get_r2_score(y_validation, validation_predictions)
		row += [
			get_r2_score_string(r2_score_training),
			get_r2_score_string(r2_score_validation),
		]
		if i == 0:
			headers += [
				"R^2 (IS)",
				"R^2 (OOS)"
			]
		table.append(row)
	print_table(table)

def get_r2_score_string(r2_score: float) -> str:
	r2_score_string = f"{r2_score:.2%}"
	if r2_score > 0.01:
		r2_score_string = f"{Fore.GREEN}{r2_score_string}{Style.RESET_ALL}"
	elif r2_score < 0:
		r2_score_string = f"{Fore.RED}{r2_score_string}{Style.RESET_ALL}"
	return r2_score_string

def get_features_labels(features: list[OhlcFeature], labels: list[float]) -> tuple[npt.NDArray, npt.NDArray]:
	feature_values = [x.values for x in features]
	x = np.array(feature_values).T
	y = np.array(labels)
	return x, y

def get_ohlc_features(
	symbol: str,
	start: pd.Timestamp,
	end: pd.Timestamp,
	filter_mode: FilterMode
) -> tuple[list[OhlcFeature], list[float]]:
	series = read_ohlc_series(symbol)
	records = series.values()

	momentum2_values = []
	momentum3_values = []
	close_open_values = []
	close_high_values = []
	close_low_values = []
	body_values = []
	high_low_values = []
	close_range_values = []
	return_values = []

	for i, record in enumerate(records[:-1]):
		if i < 5 or record.time < start:
			continue
		if record.time >= end:
			break
		if Strategy.is_banned_symbol(symbol, record.time):
			continue
		yesterday = records[i - 1]
		today = record
		tomorrow = records[i + 1]

		try:
			momentum2 = get_rate_of_change(today.close, yesterday.close)
			momentum3 = get_rate_of_change(today.close, records[i - 2].close)
			close_open = get_rate_of_change(today.close, today.open)
			close_high = get_rate_of_change(today.close, today.high)
			close_low = get_rate_of_change(today.close, today.low)
			high_low_ratio = get_rate_of_change(today.high, today.low)
			body_ratio = (today.close - today.open) / (today.high - today.low)
			close_range_ratio = (today.close - today.low) / (today.high - today.low)
			returns = get_rate_of_change(tomorrow.close, today.close)
		except ZeroDivisionError:
			continue

		match filter_mode:
			case FilterMode.POSITIVE:
				if momentum2 < 0:
					continue
			case FilterMode.NEGATIVE:
				if momentum2 > 0:
					continue
			case FilterMode.NEW_HIGH_5 | FilterMode.NEW_LOW_5:
				offset = i + 1
				recent_records = records[offset - 5:offset]
				closes = [x.close for x in recent_records]
				if filter_mode == FilterMode.NEW_HIGH_5 and record.close != max(closes):
					continue
				elif filter_mode == FilterMode.NEW_LOW_5 and record.close != min(closes):
					continue

		momentum2_values.append(momentum2)
		momentum3_values.append(momentum3)
		close_open_values.append(close_open)
		close_high_values.append(close_high)
		close_low_values.append(close_low)
		high_low_values.append(high_low_ratio)
		body_values.append(body_ratio)
		close_range_values.append(close_range_ratio)
		return_values.append(returns)

	features = [
		OhlcFeature("Momentum 2", momentum2_values),
		OhlcFeature("Momentum 3", momentum3_values),
		OhlcFeature("Close/Open Ratio", close_open_values),
		OhlcFeature("Close/High Ratio", close_high_values),
		OhlcFeature("Close/Low Ratio", close_low_values),
		OhlcFeature("High/Low Ratio", high_low_values),
		OhlcFeature("Body Ratio", body_values),
		# OhlcFeature("Close/Range Ratio", close_range_values),
	]
	return features, return_values