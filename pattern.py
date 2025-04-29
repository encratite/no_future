from functools import reduce
from statistics import mean
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from colorama import Fore, Style

from common import (
	read_ohlc_series,
	get_rate_of_change,
	print_table,
	format_percentage
)
from ohlc import OhlcRecord

type PatternFunction = Callable[[list[OhlcRecord], PatternFeatures], bool]

class PatternFeatures:
	momentum2: float
	momentum3: float
	momentum5: float
	momentum10: float
	returns: list[float]

	def __init__(self, i: int, records: list[OhlcRecord]) -> None:
		self.momentum2 = get_momentum(2, i, records)
		self.momentum3 = get_momentum(3, i, records)
		self.momentum5 = get_momentum(5, i, records)
		self.momentum10 = get_momentum(10, i, records)
		self.returns = [get_momentum(2, i - x, records) for x in range(4)]

class Pattern:
	name: str
	_long: bool
	_function: PatternFunction
	_features: list[PatternFeatures]
	_returns: list[float]
	_return_times: list[pd.Timestamp]
	_unmatched_returns: list[float]

	def __init__(self, name: str, long: bool, function: PatternFunction) -> None:
		self.name = name
		self._long = long
		self._function = function
		self._features = []
		self._returns = []
		self._return_times = []
		self._unmatched_returns = []

	def process(self, i: int, records: list[OhlcRecord]) -> None:
		offset = i + 1
		limit = 4
		local_records = list(reversed(records[offset - limit: offset]))
		features = PatternFeatures(i, records)
		match = self._function(local_records, features)
		tomorrow = records[i + 1]
		today = records[i]
		if self._long:
			returns = get_rate_of_change(tomorrow.close, today.close)
		else:
			returns = get_rate_of_change(today.close, tomorrow.close)
		if match:
			self._features.append(features)
			self._returns.append(returns)
			self._return_times.append(today.time)
		else:
			self._unmatched_returns.append(returns)

	def has_samples(self) -> bool:
		return len(self._features) > 0

	def get_returns(self) -> tuple[float, float, float]:
		total_return = 1
		for returns in self._returns:
			total_return *= 1 + returns
		total_return -= 1
		return mean(self._returns), mean(self._unmatched_returns), total_return

	def get_hit_rate(self) -> float:
		gains = [x for x in self._returns if x > 0]
		hit_rate = len(gains) / len(self._returns)
		return hit_rate

	def get_prevalence(self) -> tuple[float, int]:
		trades = len(self._features)
		prevalence = trades / (trades + len(self._unmatched_returns))
		return prevalence, trades

	def get_equity_curve(self) -> pd.DataFrame:
		equity = []
		for returns in self._returns:
			value = equity[-1] if len(equity) > 0 else 1
			new_value = value * (1 + returns)
			equity.append(new_value)
		df = pd.DataFrame({
			"time": self._return_times,
			self.name: equity
		})
		return df

def analyze_pattern(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
	assert start < end
	patterns = [
		Pattern("momentum2", True, lambda ohlc, features: features.momentum2 > 0.01),
		Pattern("-momentum2", False, lambda ohlc, features: features.momentum2 < - 0.01),
		Pattern("volume", True, lambda ohlc, features: ohlc[0].volume / (ohlc[1].volume + 1) > 2),
		Pattern("-volume", False, lambda ohlc, features: ohlc[0].volume / (ohlc[1].volume + 1) < 0.5),
		# Pattern("momentum2, momentum3", True, lambda ohlc, features: features.momentum2 > 0.005 and features.momentum3 > 0.015),
		# Pattern("-momentum2, -momentum3", False, lambda ohlc, features: features.momentum2 > - 0.005 and features.momentum3 < -0.015),
		# Pattern("011", True, lambda ohlc, features: ohlc[0].close > ohlc[1].close > ohlc[2].close < ohlc[3].close and ohlc[0].close > ohlc[3].close),
		# Pattern("100", False, lambda ohlc, features: ohlc[0].close < ohlc[1].close < ohlc[2].close > ohlc[3].close and ohlc[0].close < ohlc[3].close),
		# Pattern("000, channel", True, lambda ohlc, features: ohlc[0].close > ohlc[1].close > ohlc[2].close and ohlc[0].low > ohlc[1].low > ohlc[2].low and ohlc[0].high > ohlc[1].high > ohlc[2].high),
		# Pattern("111, -channel", False, lambda ohlc, features: ohlc[0].close < ohlc[1].close < ohlc[2].close and ohlc[0].low < ohlc[1].low < ohlc[2].low and ohlc[0].high < ohlc[1].high < ohlc[2].high),
	]
	series = read_ohlc_series(symbol)
	records: list[OhlcRecord] = series.values()
	past_offset = pd.DateOffset(years=1)
	for i, record in enumerate(records):
		if record.time < start - past_offset:
			continue
		if record.time < start:
			continue
		if record.time >= end:
			break
		for pattern in patterns:
			pattern.process(i, records)
	print_statistics(start, end, patterns)
	render_returns(patterns)

def render_returns(patterns: list[Pattern]) -> None:
	id_var = "time"
	var_name = "Strategy"
	value_name = "value_name"
	pattern_dfs = [x.get_equity_curve() for x in patterns]
	merged_df = reduce(lambda left, right: pd.merge(left, right, on=id_var, how="outer"), pattern_dfs)
	names = [x.name for x in patterns]
	melted_df = merged_df.melt(
		id_vars=id_var,
		value_vars=names,
		var_name=var_name,
		value_name=value_name
	)
	plt.figure(figsize=(12, 8))
	sns.lineplot(melted_df, x=id_var, y=value_name, hue=var_name)
	plt.xlim(melted_df[id_var].min(), melted_df[id_var].max())
	plt.xlabel("Time")
	plt.ylabel("Equity")
	plt.title(f"Equity Curve")
	plt.tight_layout()
	plt.show()
	plt.close()

def print_statistics(start: pd.Timestamp, end: pd.Timestamp, patterns: list[Pattern]) -> None:
	years = (end - start).days / 365.25
	headers = [
		"Pattern",
		"Return (Pattern)",
		"Return (Others)",
		"Annual Return",
		"Hit Rate",
		"Prevalence",
		"Trades"
	]
	table = [headers]
	for pattern in patterns:
		prevalence, trades = pattern.get_prevalence()
		if prevalence > 0.02:
			prevalence_string = f"{prevalence:.2%}"
		else:
			prevalence_string = f"{Fore.RED}{prevalence:.2%}{Style.RESET_ALL}"
		if pattern.has_samples():
			mean_return_pattern, mean_return_others, total_return = pattern.get_returns()
			mean_annual_return = total_return / years
			hit_rate = pattern.get_hit_rate()
			if hit_rate > 0.6:
				hit_rate_string = f"{Fore.GREEN}{hit_rate:.1%}{Style.RESET_ALL}"
			elif hit_rate < 0.45:
				hit_rate_string = f"{Fore.RED}{hit_rate:.1%}{Style.RESET_ALL}"
			else:
				hit_rate_string = f"{hit_rate:.1%}"
			row = [
				pattern.name,
				format_percentage(mean_return_pattern),
				format_percentage(mean_return_others),
				format_percentage(mean_annual_return),
				hit_rate_string,
				prevalence_string,
				trades
			]
		else:
			row = [
				pattern.name,
				"-",
				"-",
				"-",
				prevalence_string,
				trades
			]
		table.append(row)
	print_table(table)

def get_momentum(days: int, i: int, records: list[OhlcRecord]) -> float:
	return get_rate_of_change(records[i].close, records[i - days + 1].close)