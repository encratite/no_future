from math import sqrt
from statistics import stdev

import pandas as pd

from common import (
	read_ohlc_series,
	get_rate_of_change
)
from ohlc import OhlcRecord
from series import TimeSeries
from strategy import Strategy
from .constant import REGIME_DAYS, VOLATILITY_DAYS, DEFAULT_QUANTILES
from .data import HeatmapData
from .quantile import render_quantile_data

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
	series = read_ohlc_series(symbol)
	for x_axis, y_axis in combinations:
		render_heatmap(symbol, start, end, x_axis, y_axis, DEFAULT_QUANTILES, series, statistics_only)

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