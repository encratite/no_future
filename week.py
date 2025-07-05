import os
from typing import cast
import datetime
import calendar

import pandas as pd

from configuration import Configuration
from series import TimeSeries, OhlcRecord

def analyze_weekly_min_max(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
	path = os.path.join(Configuration.CSV_INTRADAY_DIRECTORY, f"{symbol}.csv")
	series = TimeSeries.read_ohlc_csv(path)
	records_by_week: dict[tuple[int, int], list[OhlcRecord]] = {}
	for record in series.values():
		record = cast(OhlcRecord, record)
		if record.time < start:
			continue
		if record.time >= end:
			break
		key = (record.time.year, record.time.week)
		if key in records_by_week:
			records_by_week[key].append(record)
		else:
			records_by_week[key] = [record]
	min_frequency: dict[tuple[int, datetime.time], int] = {}
	max_frequency: dict[tuple[int, datetime.time], int] = {}
	for key, records in records_by_week.items():
		min_record = min(records, key=lambda x: x.close)
		max_record = max(records, key=lambda x: x.close)
		add_record(min_record, min_frequency)
		add_record(max_record, max_frequency)
	print_frequency("Best time to buy", min_frequency)
	print_frequency("Best time to sell", max_frequency)

def add_record(record: OhlcRecord, frequency: dict[tuple[int, datetime.time], int]) -> None:
	key = (record.time.dayofweek, record.time.time())
	if key in frequency:
		frequency[key] += 1
	else:
		frequency[key] = 1

def print_frequency(description: str, frequency: dict[tuple[int, datetime.time], int]) -> None:
	pairs = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
	total_count = 0
	for _, count in pairs:
		total_count += count
	print(f"{description}:")
	for i, pair in enumerate(pairs[:10]):
		key = pair[0]
		day, time = key
		day_string = calendar.day_name[day]
		count = pair[1]
		ratio = count / total_count
		print(f"{i + 1}. {day_string} {time}: {ratio:.2%}")
	print("")

analyze_weekly_min_max("SPY", pd.Timestamp("2020-01-01"), pd.Timestamp("2025-07-05"))