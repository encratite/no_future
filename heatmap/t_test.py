import os
from math import sqrt, prod
from statistics import mean, stdev

import pandas as pd
import scipy
from colorama import Fore, Style

from common import (
	format_percentage,
	print_table
)
from configuration import Configuration
from constant import DAYS_PER_YEAR, TRADING_DAYS_PER_YEAR
from ohlc import OhlcRecord
from series import TimeSeries
from .constant import QUANTILE_RADIUS

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
	years = (end - start).days / DAYS_PER_YEAR
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
	risk_free_rate = get_risk_free_rate(start, end, series)
	mean_daily_returns = mean(daily_returns)
	daily_standard_deviation = stdev(daily_returns)
	mean_annual_returns = TRADING_DAYS_PER_YEAR * mean_daily_returns
	standard_deviation_factor = sqrt(TRADING_DAYS_PER_YEAR)
	standard_deviation = standard_deviation_factor * daily_standard_deviation
	excess_returns = mean_annual_returns - risk_free_rate
	if standard_deviation == 0:
		return "-"

	sharpe_ratio = excess_returns / standard_deviation
	return f"{sharpe_ratio:.2f}"