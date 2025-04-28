from collections import deque
from statistics import mean, stdev
from typing import Callable, Final
from colorama import Fore, Style

import pandas as pd

from common import (
	read_ohlc_series,
	get_rate_of_change,
	print_table,
	format_percentage
)
from ohlc import OhlcRecord
from strategy import Strategy

type PatternFunction = Callable[[list[OhlcRecord], PatternFeatures], bool]

class ZScoreBuffer:
	BUFFER_SIZE: Final[int] = 252

	_buffer: deque[float]

	def __init__(self) -> None:
		self._buffer = deque()

	def add(self, value: float) -> None:
		self._buffer.append(value)
		while len(self._buffer) > self.BUFFER_SIZE:
			self._buffer.popleft()

	def z_score(self) -> float:
		most_recent_value = self._buffer[-1]
		z_score = (most_recent_value - mean(self._buffer)) / stdev(self._buffer)
		return z_score

class ZScoreBuffers:
	momentum2: ZScoreBuffer
	momentum3: ZScoreBuffer
	momentum5: ZScoreBuffer
	momentum10: ZScoreBuffer

	def __init__(self):
		self.momentum2 = ZScoreBuffer()
		self.momentum3 = ZScoreBuffer()
		self.momentum5 = ZScoreBuffer()
		self.momentum10 = ZScoreBuffer()

class PatternFeatures:
	momentum2_zscore: float
	momentum3_zscore: float
	momentum5_zscore: float
	momentum10_zscore: float

	def __init__(self, buffers: ZScoreBuffers):
		self.momentum2_zscore = buffers.momentum2.z_score()
		self.momentum3_zscore = buffers.momentum3.z_score()
		self.momentum5_zscore = buffers.momentum5.z_score()
		self.momentum10_zscore = buffers.momentum10.z_score()

class Pattern:
	name: str
	_function: PatternFunction
	_features: list[PatternFeatures]
	_returns: list[float]
	_unmatched_returns: list[float]

	def __init__(self, name: str, function: PatternFunction) -> None:
		self.name = name
		self._function = function
		self._features = []
		self._returns = []
		self._unmatched_returns = []

	def process(self, i: int, records: list[OhlcRecord], buffers: ZScoreBuffers) -> None:
		offset = i + 1
		limit = 4
		local_records = records[offset - limit: offset]
		features = PatternFeatures(buffers)
		match = self._function(local_records, features)
		returns = get_rate_of_change(records[i + 1].close, records[i].close)
		if match:
			self._features.append(features)
			self._returns.append(returns)
		else:
			self._unmatched_returns.append(returns)

	def has_samples(self) -> bool:
		return len(self._features) > 0

	def get_mean_returns(self) -> tuple[float, float]:
		return mean(self._returns), mean(self._unmatched_returns)

	def get_hit_rate(self, short: bool) -> float:
		gains = [x for x in self._returns if x > 0]
		hit_rate = len(gains) / len(self._returns)
		if short:
			hit_rate = 1 - hit_rate
		return hit_rate

	def get_prevalence(self) -> float:
		return len(self._features) / (len(self._features) + len(self._unmatched_returns))

def analyze_pattern(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
	assert start < end
	patterns = [
		Pattern("momentum2", lambda ohlc, features: features.momentum2_zscore > 1.5),
		Pattern("-momentum2", lambda ohlc, features: features.momentum2_zscore < -1.5),
		Pattern("momentum2, momentum3", lambda ohlc, features: features.momentum2_zscore > 1.5 and features.momentum3_zscore > 1.5),
		Pattern("-momentum2, -momentum3", lambda ohlc, features: features.momentum2_zscore < -1.5 and features.momentum3_zscore < -1.5),
		Pattern("momentum2, momentum3, momentum5", lambda ohlc, features: features.momentum2_zscore > 1 and features.momentum3_zscore > 0.5 and features.momentum5_zscore > 0.5),
		Pattern("-momentum2, -momentum3, -momentum5", lambda ohlc, features: features.momentum2_zscore < -1 and features.momentum3_zscore < -0.5 and features.momentum5_zscore < -0.5),
		Pattern("011, momentum2", lambda ohlc, features: ohlc[0].close > ohlc[1].close < ohlc[2].close and features.momentum2_zscore > 1.5),
		Pattern("100, -momentum2", lambda ohlc, features: ohlc[0].close < ohlc[1].close > ohlc[2].close and features.momentum2_zscore < -1.5),
		Pattern("011, high, momentum2", lambda ohlc, features: ohlc[0].close > ohlc[1].close < ohlc[2].close and ohlc[0].high < ohlc[2].high and ohlc[1].high < ohlc[2].high and features.momentum2_zscore > 1.5),
		Pattern("100, low, -momentum2", lambda ohlc, features: ohlc[0].close < ohlc[1].close > ohlc[2].close and ohlc[0].low > ohlc[2].low and ohlc[1].low > ohlc[2].low and features.momentum2_zscore < -1.5),
		Pattern("000, momentum2", lambda ohlc, features: ohlc[0].close > ohlc[1].close > ohlc[2].close and features.momentum2_zscore < -1),
		Pattern("111, momentum2", lambda ohlc, features: ohlc[0].close < ohlc[1].close < ohlc[2].close and features.momentum2_zscore > 1),
		Pattern("000, momentum3", lambda ohlc, features: ohlc[0].close > ohlc[1].close > ohlc[2].close and features.momentum3_zscore < -1),
		Pattern("111, momentum3", lambda ohlc, features: ohlc[0].close < ohlc[1].close < ohlc[2].close and features.momentum3_zscore > 1),
		Pattern("000, momentum5", lambda ohlc, features: ohlc[0].close > ohlc[1].close > ohlc[2].close and features.momentum5_zscore < -1),
		Pattern("111, momentum5", lambda ohlc, features: ohlc[0].close < ohlc[1].close < ohlc[2].close and features.momentum5_zscore > 1),
		Pattern("000, momentum10", lambda ohlc, features: ohlc[0].close > ohlc[1].close > ohlc[2].close and features.momentum10_zscore < -1),
		Pattern("111, momentum10", lambda ohlc, features: ohlc[0].close < ohlc[1].close < ohlc[2].close and features.momentum10_zscore > 1),
		Pattern("000, channel", lambda ohlc, features: ohlc[0].close > ohlc[1].close > ohlc[2].close and ohlc[0].low > ohlc[1].low > ohlc[2].low and ohlc[0].high > ohlc[1].high > ohlc[2].high and features.momentum2_zscore < -1),
		Pattern("111, -channel", lambda ohlc, features: ohlc[0].close < ohlc[1].close < ohlc[2].close and ohlc[0].low < ohlc[1].low < ohlc[2].low and ohlc[0].high < ohlc[1].high < ohlc[2].high and features.momentum2_zscore > 1),
	]
	series = read_ohlc_series(symbol)
	records: list[OhlcRecord] = series.values()
	buffers = ZScoreBuffers()
	past_offset = pd.Timedelta(days=10 + 1)
	for i, record in enumerate(records):
		lookback = record.time - past_offset
		if record.time < lookback or Strategy.is_banned_symbol(symbol, record.time) or Strategy.is_banned_symbol(symbol, lookback):
			continue
		momentum2 = get_momentum(2, i, records)
		momentum3 = get_momentum(3, i, records)
		momentum5 = get_momentum(5, i, records)
		momentum10 = get_momentum(10, i, records)
		buffers.momentum2.add(momentum2)
		buffers.momentum3.add(momentum3)
		buffers.momentum5.add(momentum5)
		buffers.momentum10.add(momentum10)
		if record.time < start or record.time >= end:
			continue
		for pattern in patterns:
			pattern.process(i, records, buffers)
	print_statistics(patterns)

def print_statistics(patterns: list[Pattern]) -> None:
	headers = [
		"Pattern",
		"Return (Pattern)",
		"Return (Others)",
		"Hit Rate",
		"Prevalence"
	]
	table = [headers]
	for pattern in patterns:
		prevalence = pattern.get_prevalence()
		if prevalence > 0.02:
			prevalence_string = f"{prevalence:.2%}"
		else:
			prevalence_string = f"{Fore.RED}{prevalence:.2%}{Style.RESET_ALL}"
		if pattern.has_samples():
			mean_return_pattern, mean_return_others = pattern.get_mean_returns()
			short = mean_return_pattern < 0
			hit_rate = pattern.get_hit_rate(short)
			if hit_rate > 0.6:
				hit_rate_string = f"{Fore.GREEN}{hit_rate:.1%}{Style.RESET_ALL}"
			elif hit_rate < 0.48:
				hit_rate_string = f"{Fore.RED}{hit_rate:.1%}{Style.RESET_ALL}"
			else:
				hit_rate_string = f"{hit_rate:.1%}"
			row = [
				pattern.name,
				format_percentage(mean_return_pattern),
				format_percentage(mean_return_others),
				hit_rate_string,
				prevalence_string
			]
		else:
			row = [
				pattern.name,
				"-",
				"-",
				"-",
				prevalence_string
			]
		table.append(row)
	print_table(table)

def get_momentum(days: int, i: int, records: list[OhlcRecord]) -> float:
	return get_rate_of_change(records[i].close, records[i - days + 1].close)