import calendar
from collections import defaultdict
from statistics import mean
from typing import Final, cast

import pandas as pd

from common import format_percentage, print_table, read_ohlc_series, get_rate_of_change

TABLE_CONFIGS = [
	("Symbol (Old)   ", lambda x: x.old_stats),
	("Symbol (Recent)", lambda x: x.recent_stats),
]

class SeasonalityStats:
	day_of_week_returns: defaultdict[int, list[float]]
	monthly_returns: defaultdict[int, list[float]]
	early_monthly_returns: list[float]
	late_monthly_returns: list[float]

	def __init__(self):
		self.day_of_week_returns = defaultdict(list)
		self.monthly_returns = defaultdict(list)
		self.early_monthly_returns = []
		self.late_monthly_returns = []

	def add(self, time: pd.Timestamp, returns: float):
		self.day_of_week_returns[time.day_of_week].append(returns)
		self.monthly_returns[time.month].append(returns)
		early_late_delta = pd.Timedelta(days=1)
		one_day = pd.Timedelta(days=1)
		previous_time: pd.Timestamp = time - early_late_delta
		saturday: Final[int] = 5
		while cast(int, previous_time.day_of_week) >= saturday:
			previous_time -= one_day
		next_time: pd.Timestamp = time + early_late_delta
		while next_time.day_of_week >= saturday:
			next_time += one_day
		if previous_time.month != time.month:
			self.early_monthly_returns.append(returns)
		elif next_time.month != time.month:
			self.late_monthly_returns.append(returns)

class SymbolStats:
	symbol: str
	old_stats: SeasonalityStats
	recent_stats: SeasonalityStats

	def __init__(self, symbol: str):
		self.symbol = symbol
		self.old_stats = SeasonalityStats()
		self.recent_stats = SeasonalityStats()

def analyze_seasonality(symbols: list[str], start: pd.Timestamp, split: pd.Timestamp, end: pd.Timestamp) -> None:
	assert start < split < end
	stats = [analyze_seasonality_by_symbol(symbol, start, split, end) for symbol in symbols]
	print_day_of_week_stats(stats)
	print_monthly_stats(stats)
	print_early_late_monthly_stats(stats)

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

def print_early_late_monthly_stats(stats: list[SymbolStats]) -> None:
	for title, sample_stats_fn in TABLE_CONFIGS:
		month_headers = [title, "Turn of the Month (Early)", "Turn of the Month (Late)"]
		table = [month_headers]
		for symbol_stats in stats:
			sample_stats = sample_stats_fn(symbol_stats)
			early_returns = mean(sample_stats.early_monthly_returns)
			early_string = format_percentage(early_returns)
			late_returns = mean(sample_stats.late_monthly_returns)
			late_string = format_percentage(late_returns)
			row = [symbol_stats.symbol, early_string, late_string]
			table.append(row)
		print_table(table)

def analyze_seasonality_by_symbol(symbol: str, start: pd.Timestamp, split: pd.Timestamp, end: pd.Timestamp) -> SymbolStats:
	ohlc_series = read_ohlc_series(symbol)
	previous_record = None
	symbol_stats = SymbolStats(symbol)
	for time in ohlc_series:
		if time < start or time >= end:
			continue
		record = ohlc_series.get(time)
		if previous_record is not None:
			returns = get_rate_of_change(record.close, previous_record.close)
			stats = symbol_stats.old_stats if time < split else symbol_stats.recent_stats
			stats.add(time, returns)
		previous_record = record
	return symbol_stats