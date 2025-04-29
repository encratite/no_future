import warnings
from collections import defaultdict
from math import sqrt
from statistics import mean, stdev
from typing import Final

import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import QuantileTransformer

from backtest.interface import BacktestInterface
from common import get_rate_of_change
from configuration import Configuration
from constant import TRADING_DAYS_PER_YEAR, DAYS_PER_YEAR
from ohlc import OhlcRecord
from .base import Strategy

class QuantileFeatures:
	MOMENTUM2: Final[str] = "momentum2"
	MOMENTUM3: Final[str] = "momentum3"
	MOMENTUM10: Final[str] = "momentum10"
	REGIME: Final[str] = "regime"
	VOLATILITY: Final[str] = "volatility"
	VOLUME: Final[str] = "volume"
	OPEN_INTEREST: Final[str] = "open_interest"

class QuantileRadiusStrategy(Strategy):
	STATISTICAL_WINDOW_SIZE: Final[int] = 6 * TRADING_DAYS_PER_YEAR
	FEATURE_WINDOW_SIZE: Final[int] = TRADING_DAYS_PER_YEAR
	UPDATE_INTERVAL: Final[int] = round(DAYS_PER_YEAR)
	T_STATISTIC_MINIMUM: Final[float] = 2.0
	VOLATILITY_DAYS: Final[int] = 20

	_symbol: str
	_feature1: str
	_feature2: str
	_radius: float

	_a: float | None
	_b: float | None
	_transformer1: QuantileTransformer | None
	_transformer2: QuantileTransformer | None
	_signal: int | None
	_last_update: pd.Timestamp | None

	def __init__(self, symbol: str, feature1: str, feature2: str, radius: float) -> None:
		super().__init__(f"Quantile Radius ({symbol}, {feature1}, {feature2}, {radius:.2f})")
		assert 0 < radius < 1
		self._symbol = symbol
		self._feature1 = feature1
		self._feature2 = feature2
		self._radius = radius
		self.reset()

	def get_signals(self, interface: BacktestInterface) -> dict[str, float]:
		if self._last_update is None or (interface.time - self._last_update) >= pd.Timedelta(days=self.UPDATE_INTERVAL):
			self._update(interface)
		if self._a is not None and self._b is not None:
			feature_window = interface.get_records(self._symbol, count=self.FEATURE_WINDOW_SIZE)
			features = self._get_features(feature_window)
			feature1 = features[self._feature1]
			feature2 = features[self._feature2]
			quantile1 = self._transformer1.transform([[feature1]])
			quantile1 = quantile1[0, 0].item()
			quantile2 = self._transformer2.transform([[feature2]])
			quantile2 = quantile2[0, 0].item()
			if self._in_range(quantile1, quantile2, self._a, self._b):
				return {
					self._symbol: self._signal
				}
			else:
				return {}
		else:
			return {}

	def reset(self) -> None:
		self._a = None
		self._b = None
		self._transformer1 = None
		self._transformer2 = None
		self._signal = None
		self._last_update = None

	def _update(self, interface: BacktestInterface) -> None:
		warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing")
		records = interface.get_records(self._symbol, count=self.STATISTICAL_WINDOW_SIZE)
		all_features: defaultdict[str, list[float]] = defaultdict(list)
		labels: list[float] = []
		for i, tomorrow in enumerate(records):
			offset = i + 1
			feature_window = records[offset : offset + self.FEATURE_WINDOW_SIZE]
			if len(feature_window) < self.FEATURE_WINDOW_SIZE:
				break
			label = get_rate_of_change(tomorrow.close, records[offset].close)
			features = self._get_features(feature_window)
			for key, value in features.items():
				all_features[key].append(value)
			labels.append(label)
		transformer1 = QuantileTransformer()
		transformer2 = QuantileTransformer()
		features1 = all_features[self._feature1]
		features2 = all_features[self._feature2]
		features_array1 = np.array(features1)
		features_array1 = features_array1.reshape(-1, 1)
		features_array2 = np.array(features2)
		features_array2 = features_array2.reshape(-1, 1)
		quantiles1 = transformer1.fit_transform(features_array1)
		quantiles1 = quantiles1.squeeze()
		quantiles2 = transformer2.fit_transform(features_array2)
		quantiles2 = quantiles2.squeeze()
		corners = [
			(0, 0),
			(0, 1),
			(1, 0),
			(1, 1)
		]
		corner_stats = []
		for a, b in corners:
			positive = []
			negative = []
			for x, y, returns in zip(quantiles1, quantiles2, labels):
				if self._in_range(x, y, a, b):
					positive.append(returns)
				else:
					negative.append(returns)
			if len(positive) == 0 or len(negative) == 0:
				continue
			if mean(positive) > mean(negative):
				hypothesis = "greater"
				signal = 1
			else:
				hypothesis = "less"
				signal = -1
			statistic = scipy.stats.ttest_ind(
				a=positive,
				b=negative,
				equal_var=False,
				nan_policy="raise",
				random_state=Configuration.SEED,
				alternative=hypothesis
			)
			corner_stats.append((a, b, statistic, signal))
		corner_stats = sorted(corner_stats, key=lambda t: abs(t[2].statistic), reverse=True)
		a, b, statistic, signal = corner_stats[0]
		if abs(statistic.statistic) > self.T_STATISTIC_MINIMUM:
			self._a = a
			self._b = b
			self._transformer1 = transformer1
			self._transformer2 = transformer2
			self._signal = signal
			print(f"[{interface.time}] Active: a = {a}, b = {b}, signal = {signal}, t = {statistic.statistic:.2f}")
		else:
			print(f"[{interface.time}] Failed: a = {a}, b = {b}, signal = {signal}, t = {statistic.statistic:.2f}")
			self.reset()
		self._last_update = interface.time

	def _in_range(self, x: float, y: float, a: float, b: float) -> bool:
		return (x - a) ** 2 + (y - b) ** 2 < self._radius ** 2

	@staticmethod
	def _get_features(records: list[OhlcRecord]) -> dict[str, float]:
		momentum2 = QuantileRadiusStrategy._get_momentum(2, records)
		momentum3 = QuantileRadiusStrategy._get_momentum(3, records)
		momentum10 = QuantileRadiusStrategy._get_momentum(10, records)
		regime = get_rate_of_change(records[0].close, records[-1].close)
		closes = [x.close for x in records[:QuantileRadiusStrategy.VOLATILITY_DAYS + 1]]
		returns = [get_rate_of_change(a, b) for a, b in zip(closes[1:], closes)]
		volatility = stdev(returns) * sqrt(len(returns))
		previous_volume = records[1].volume
		if previous_volume == 0:
			previous_volume = 1
		volume = get_rate_of_change(records[0].volume, previous_volume)
		previous_open_interest = records[1].open_interest
		if previous_open_interest == 0:
			previous_open_interest = 1
		open_interest = get_rate_of_change(records[0].open_interest, previous_open_interest)
		features = {
			QuantileFeatures.MOMENTUM2: momentum2,
			QuantileFeatures.MOMENTUM3: momentum3,
			QuantileFeatures.MOMENTUM10: momentum10,
			QuantileFeatures.REGIME: regime,
			QuantileFeatures.VOLATILITY: volatility,
			QuantileFeatures.VOLUME: volume,
			QuantileFeatures.OPEN_INTEREST: open_interest,
		}
		return features

	@staticmethod
	def _get_momentum(days: int, records: list[OhlcRecord]) -> float:
		momentum = get_rate_of_change(records[0].close, records[days - 1].close)
		return momentum