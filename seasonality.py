import calendar
import os
from collections import defaultdict
from itertools import repeat
from statistics import mean

import pandas as pd

from common import execute_thread_pool, format_percentage, print_table
from configuration import Configuration
from series import TimeSeries

TABLE_CONFIGS = [
	("Symbol (IS) ", lambda x: x.in_sample_stats),
	("Symbol (OOS)", lambda x: x.out_of_sample_stats),
]

class SeasonalityStats:
	day_of_week_returns: defaultdict[int, list[float]]
	monthly_returns: defaultdict[int, list[float]]

	def __init__(self):
		self.day_of_week_returns = defaultdict(list)
		self.monthly_returns = defaultdict(list)

	def add(self, time: pd.Timestamp, returns: float):
		self.day_of_week_returns[time.day_of_week].append(returns)
		self.monthly_returns[time.month].append(returns)

class SymbolStats:
	symbol: str
	in_sample_stats: SeasonalityStats
	out_of_sample_stats: SeasonalityStats

	def __init__(self, symbol: str):
		self.symbol = symbol
		self.in_sample_stats = SeasonalityStats()
		self.out_of_sample_stats = SeasonalityStats()

def analyze_seasonality(symbols: list[str], start: pd.Timestamp, split: pd.Timestamp, end: pd.Timestamp) -> None:
	assert start < split < end
	stats_iterable = execute_thread_pool(analyze_seasonality_by_symbol, symbols, repeat(start), repeat(split), repeat(end))
	stats = list(stats_iterable)
	print_day_of_week_stats(stats)
	print_monthly_stats(stats)

def print_day_of_week_stats(stats: list[SymbolStats]) -> None:
	week_range = range(5)
	days_of_the_week = [calendar.day_name[i] for i in week_range]
	for title, sample_stats_fn in TABLE_CONFIGS:
		headers = [title] + days_of_the_week
		table = [headers]
		for symbol_stats in stats:
			row = [symbol_stats.symbol]
			sample_stats = sample_stats_fn(symbol_stats)
			for i in week_range:
				day_of_week_returns = sample_stats.day_of_week_returns[i]
				mean_returns = mean(day_of_week_returns)
				performance_string = format_percentage(mean_returns)
				row.append(performance_string)
			table.append(row)
		print_table(table)

def print_monthly_stats(stats: list[SymbolStats]) -> None:
	month_range = [i + 1 for i in range(12)]
	month_names = [calendar.month_name[i] for i in month_range]
	for title, sample_stats_fn in TABLE_CONFIGS:
		month_headers = [title] + month_names
		table = [month_headers]
		for symbol_stats in stats:
			row = [symbol_stats.symbol]
			sample_stats = sample_stats_fn(symbol_stats)
			for i in month_range:
				monthly_returns = sample_stats.monthly_returns[i]
				mean_returns = mean(monthly_returns)
				performance_string = format_percentage(mean_returns)
				row.append(performance_string)
			table.append(row)
		print_table(table)

def analyze_seasonality_by_symbol(symbol: str, start: pd.Timestamp, split: pd.Timestamp, end: pd.Timestamp) -> SymbolStats:
	if "." not in symbol:
		file_name = f"{symbol}.F1"
	else:
		file_name = symbol
	path = os.path.join(Configuration.FEATHER_DIRECTORY, f"{file_name}.feather")
	ohlc_series = TimeSeries.read_ohlc_feather(path)
	previous_record = None
	symbol_stats = SymbolStats(symbol)
	for time in ohlc_series:
		if time < start or time >= end:
			continue
		record = ohlc_series.get(time)
		if previous_record is not None:
			returns = get_rate_of_change(record.close, previous_record.close)
			stats = symbol_stats.in_sample_stats if time < split else symbol_stats.out_of_sample_stats
			stats.add(time, returns)
		previous_record = record
	return symbol_stats

def get_rate_of_change(new_value: float, old_value: float) -> float:
	return new_value / old_value - 1