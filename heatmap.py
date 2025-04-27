import os
import warnings
from math import sqrt, prod
from statistics import mean, stdev
from typing import Final, Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import seaborn as sns
from colorama import Fore, Style
from sklearn.preprocessing import quantile_transform

from common import (
	read_ohlc_series,
	get_rate_of_change,
	format_percentage,
	print_table
)
from configuration import Configuration
from ohlc import OhlcRecord
from series import TimeSeries
from strategy import Strategy

VOLATILITY_DAYS: Final[int] = 20
REGIME_DAYS: Final[int] = 200
DAYS_PER_YEAR: Final[float] = 365.25
QUANTILE_RADIUS: Final[float] = 0.25

class HeatmapFeature:
	id_: str
	description: str
	values: list[float]

	def __init__(self, id_: str, description: str):
		self.id_ = id_
		self.description = description
		self.values = []

	def append(self, value: float) -> None:
		self.values.append(value)

class HeatmapData:
	y_values: list[float]
	momentum2: HeatmapFeature
	momentum3: HeatmapFeature
	momentum10: HeatmapFeature
	regime: HeatmapFeature
	volume: HeatmapFeature
	open_interest: HeatmapFeature
	volatility: HeatmapFeature

	def __init__(self):
		self.y_values = []
		self.momentum2 = HeatmapFeature("momentum2", "Momentum (2 Days)")
		self.momentum3 = HeatmapFeature("momentum3", "Momentum (3 Days)")
		self.momentum10 = HeatmapFeature("momentum10", f"Momentum (10 Days)")
		self.regime = HeatmapFeature("regime", "Regime")
		self.volume = HeatmapFeature("volume", "Change in volume from yesterday")
		self.open_interest = HeatmapFeature("interest", "Change in open interest from yesterday")
		self.volatility = HeatmapFeature("volatility", f"Volatility ({VOLATILITY_DAYS} Days)")

	def get_values(self) -> dict[str, tuple[str, Any]]:
		output = {}
		features = [
			self.momentum2,
			self.momentum3,
			self.momentum10,
			self.regime,
			self.volume,
			self.open_interest,
			self.volatility
		]
		for feature in features:
			quantile_values = get_quantile_transform(feature.values)
			output[feature.id_] = (feature.description, quantile_values)
		return output

def render_heatmap_all(symbol: str, start: pd.Timestamp, end: pd.Timestamp, statistics_only: bool) -> None:
	combinations = [
		("momentum2", "momentum3"),
		("momentum2", "momentum10"),
		("momentum2", "regime"),
		("momentum2", "volume"),
		("momentum2", "interest"),
		("momentum2", "volatility"),
		# ("momentum3", "volume"),
		# ("momentum3", "interest"),
		# ("momentum3", "volatility"),
		# ("volume", "interest"),
		# ("volume", "volatility"),
	]
	quantiles = 5
	series = read_ohlc_series(symbol)
	for x_axis, y_axis in combinations:
		render_heatmap(symbol, start, end, x_axis, y_axis, quantiles, series, statistics_only)

def render_heatmap(
	symbol: str,
	start: pd.Timestamp,
	end: pd.Timestamp,
	x_axis: str,
	y_axis: str,
	quantiles: int,
	series: TimeSeries[OhlcRecord] | None = None,
	statistics_only: bool = False
) -> None:
	assert start < end
	assert quantiles >= 2
	if series is None:
		series = read_ohlc_series(symbol)
	data = HeatmapData()
	time = start
	first_record_time = next(iter(series))
	if time < first_record_time:
		time = first_record_time + pd.Timedelta(days=30)
		print(f"Warning: adjusted start time to {time} due to missing records")
	one_day = pd.Timedelta(days=1)
	while time < end:
		if Strategy.is_banned_symbol(symbol, time):
			time += one_day
			continue
		tomorrow = series.get(time + one_day, right=True)
		records = series.get(time, count=REGIME_DAYS + 1)
		today = records[0]
		yesterday = records[1]
		y = get_rate_of_change(tomorrow.close, today.close)
		returns = [get_rate_of_change(a.close, b.close) for a, b in zip(records[1:], records)]
		momentum2 = returns[0]
		momentum3 = get_rate_of_change(today.close, records[2].close)
		momentum10 = get_rate_of_change(records[0].close, records[10 - 1].close)
		regime = get_rate_of_change(records[0].close, records[REGIME_DAYS - 1].close)
		# Workaround for zero volume record in SI
		volume = get_rate_of_change(today.volume, max(yesterday.volume, 1))
		# Workaround for zero open interest record in 6E
		open_interest = get_rate_of_change(today.open_interest, max(yesterday.open_interest, 1))
		volatility = stdev(returns[:VOLATILITY_DAYS]) * sqrt(VOLATILITY_DAYS)
		data.y_values.append(y)
		data.momentum2.append(momentum2)
		data.momentum3.append(momentum3)
		data.momentum10.append(momentum10)
		data.regime.append(regime)
		data.volume.append(volume)
		data.open_interest.append(open_interest)
		data.volatility.append(volatility)
		time += one_day
	render_quantile_data(
		symbol,
		start,
		end,
		x_axis,
		y_axis,
		quantiles,
		statistics_only,
		data,
		series
	)

def render_quantile_data(
	symbol: str,
	start: pd.Timestamp,
	end: pd.Timestamp,
	x_axis: str,
	y_axis: str,
	quantiles: int,
	statistics_only: bool,
	data: HeatmapData,
	series: TimeSeries[OhlcRecord]
) -> None:
	values = data.get_values()
	x_axis_title, x_axis_values = values[x_axis]
	y_axis_title, y_axis_values = values[y_axis]
	perform_t_test(x_axis, y_axis, x_axis_values, y_axis_values, data.y_values, series, start, end)
	if statistics_only:
		return
	mean_returns_matrix = np.zeros((quantiles, quantiles))
	annotations = np.empty((quantiles, quantiles), dtype=object)
	for i in range(quantiles):
		for j in range(quantiles):
			x_quantile_min, x_quantile_max = get_quantile_limits(i, quantiles)
			y_quantile_min, y_quantile_max = get_quantile_limits(j, quantiles)
			matching_y_values = []
			for k, y in enumerate(data.y_values):
				x_quantile = x_axis_values[k]
				y_quantile = y_axis_values[k]
				x_match = x_quantile_min < x_quantile < x_quantile_max
				y_match = y_quantile_min < y_quantile < y_quantile_max
				if x_match and y_match:
					matching_y_values.append(y)
			if len(matching_y_values) > 0:
				mean_returns = mean(matching_y_values)
			else:
				mean_returns = 0
			y_values_array = np.array(matching_y_values)
			skew = scipy.stats.skew(y_values_array)
			kurtosis = scipy.stats.kurtosis(y_values_array)
			mean_returns_matrix[i, j] = mean_returns
			trades = len(matching_y_values)
			years_traded = (end - start).days / DAYS_PER_YEAR
			trades_per_year = trades / years_traded
			if trades > 0:
				annotations[i, j] = f"Mean: {mean_returns:+.2%}\nSkew: {skew:.2f}\nKurtosis: {kurtosis:.2f}\nTrades: {trades_per_year:.1f}"
			else:
				annotations[i, j] = f"No samples"
	plt.figure(figsize=(12, 8))
	tick_labels = [f"Quantile {i + 1}" for i in range(quantiles)]
	ax = sns.heatmap(mean_returns_matrix, annot=annotations, fmt="", xticklabels=tick_labels, yticklabels=tick_labels)
	cbar = ax.collections[0].colorbar
	formatter = ticker.FuncFormatter(lambda x, _: f"{x * 100:.2f}%")
	cbar.ax.yaxis.set_major_formatter(formatter)
	plt.title(f"Single Day Returns of {symbol} by Feature Quantiles\n(from {format_date(start)} to {format_date(end)})")
	plt.xlabel(x_axis_title)
	plt.ylabel(y_axis_title)
	plt.show()
	plt.close()

def format_date(time: pd.Timestamp) -> str:
	return time.strftime("%Y-%m-%d")

def get_quantile_limits(i: int, quantiles: int) -> tuple[float, float]:
	quantile_min = i / float(quantiles)
	quantile_max = (i + 1) / float(quantiles)
	return quantile_min, quantile_max

def get_quantile_transform(values: list[float]) -> npt.NDArray:
	warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing")
	array = np.array(values).reshape(-1, 1)
	output = quantile_transform(array)
	return output

def perform_t_test(
	x_axis: str,
	y_axis: str,
	x_values: list[float],
	y_values: list[float],
	returns_values: list[float],
	series: TimeSeries[OhlcRecord],
	start: pd.Timestamp,
	end: pd.Timestamp
) -> None:
	corners = [
		(0, 0),
		(0, 1),
		(1, 0),
		(1, 1)
	]
	padding = 10
	headers = [
		x_axis.ljust(padding),
		y_axis.ljust(padding),
		"Hypothesis",
		"t-statistic",
		"p-value",
		"Mean (positive)",
		"Mean (negative)",
		"Return (positive)",
		"Return (negative)",
		"SR (positive)",
		"SR (negative)",
		"Positive samples"
	]
	table = [headers]
	for a, b in corners:
		positive = []
		negative = []
		for x, y, returns in zip(x_values, y_values, returns_values):
			if (x - (1 - a)) ** 2 + (y - (1 - b)) ** 2 < QUANTILE_RADIUS ** 2:
				positive.append(returns)
			else:
				negative.append(returns)
		greater_statistic = scipy.stats.ttest_ind(
			a=positive,
			b=negative,
			equal_var=False,
			nan_policy="raise",
			random_state=Configuration.SEED,
			alternative="greater"
		)
		less_statistic = scipy.stats.ttest_ind(
			a=positive,
			b=negative,
			equal_var=False,
			nan_policy="raise",
			random_state=Configuration.SEED,
			alternative="less"
		)
		statistics = [
			(greater_statistic, "Greater", 1),
			(less_statistic, "Less", -1)
		]
		best_statistic, best_hypothesis, returns_multiplier = min(statistics, key=lambda t: t[0].pvalue)
		if len(positive) > 0:
			mean_positive = mean(positive)
		else:
			mean_positive = 0
		if len(negative) > 0:
			mean_negative = mean(negative)
		else:
			mean_negative = 0
		positive_returns = [returns_multiplier * x for x in positive]
		negative_returns = [returns_multiplier * x for x in negative]
		positive_balance = len(positive) / len(x_values)
		row = [
			a,
			b,
			best_hypothesis,
			format_t_statistic(best_statistic.statistic),
			format_p_value(best_statistic.pvalue),
			format_percentage(mean_positive),
			format_percentage(mean_negative),
			get_mean_annual_returns(positive, start, end),
			get_mean_annual_returns(negative, start, end),
			get_sharpe_ratio(positive_returns, start, end, series),
			get_sharpe_ratio(negative_returns, start, end, series),
			f"{positive_balance:.2%}"
		]
		table.append(row)
	print_table(table, always_right=True)

def format_t_statistic(statistic: float) -> str:
	if abs(statistic) > 2.0:
		return f"{Fore.GREEN}{statistic:.3f}{Style.RESET_ALL}"
	else:
		return f"{statistic:.3f}"

def format_p_value(statistic: float) -> str:
	if statistic < 0.025:
		return f"{Fore.GREEN}{statistic:.3f}{Style.RESET_ALL}"
	else:
		return f"{statistic:.3f}"

def get_mean_annual_returns(class_returns: list[float], start: pd.Timestamp, end: pd.Timestamp) -> str:
	total = prod([x + 1 for x in class_returns])
	years = (end - start).days / 365.25
	mean_annual_returns = format_percentage((total - 1) / years)
	return mean_annual_returns

def get_risk_free_rate(start: pd.Timestamp, end: pd.Timestamp, series: TimeSeries[OhlcRecord]) -> float:
	t_bills_path = os.path.join(Configuration.FRED_DIRECTORY, "TB3MS.csv")
	t_bills = TimeSeries.read_csv(t_bills_path, True)
	risk_free_rate_values = []
	time_range = [x for x in series if start <= x < end]
	for time in time_range:
		risk_free_rate = t_bills.get(time) / 100
		risk_free_rate_values.append(risk_free_rate)
	mean_risk_free_rate = mean(risk_free_rate_values)
	return mean_risk_free_rate

def get_sharpe_ratio(daily_returns: list[float], start: pd.Timestamp, end: pd.Timestamp, series: TimeSeries[OhlcRecord]) -> str:
	trading_days_per_year: Final[int] = 252

	risk_free_rate = get_risk_free_rate(start, end, series)
	mean_daily_returns = mean(daily_returns)
	daily_standard_deviation = stdev(daily_returns)
	mean_annual_returns = trading_days_per_year * mean_daily_returns
	standard_deviation_factor = sqrt(trading_days_per_year)
	standard_deviation = standard_deviation_factor * daily_standard_deviation
	excess_returns = mean_annual_returns - risk_free_rate
	if standard_deviation == 0:
		return "-"

	sharpe_ratio = excess_returns / standard_deviation
	return f"{sharpe_ratio:.2f}"