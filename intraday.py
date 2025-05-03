import datetime as dt
from statistics import mean
from typing import cast

import pandas as pd

from common import (
	read_ohlc_series,
	get_log_returns,
	format_percentage
)
from ohlc import OhlcRecord

def analyze_session_returns(symbol: str, session_start: dt.time, session_end: dt.time, start: pd.Timestamp, end: pd.Timestamp) -> None:
	series = read_ohlc_series(symbol, intraday=True)
	records = series.values()
	sessions = []
	session_open: float | None = None
	session_end_time: pd.Timestamp | None = None
	new_day = True
	for previous_record, record in zip(records, records[1:]):
		previous_record = cast(OhlcRecord, previous_record)
		record = cast(OhlcRecord, record)
		if record.time < start or record.time >= end:
			continue
		if session_end_time is None:
			if previous_record.time.dayofweek != record.time.dayofweek:
				new_day = True
			if record.time.time() >= session_start and new_day:
				# print(f"New session at {record.time}")
				combined_date = record.time.date()
				if session_start > session_end:
					combined_date += dt.timedelta(days=1)
				combined = dt.datetime.combine(combined_date, session_end)
				session_open = record.open
				session_end_time = pd.Timestamp(combined)
		else:
			if record.time >= session_end_time:
				# print(f"End of session at {record.time}")
				session_return = get_log_returns(record.close, session_open)
				sessions.append(session_return)
				session_open = None
				session_end_time = None
				if session_start < session_end:
					new_day = False
	mean_returns = mean(sessions)
	positive_rate = len([x for x in sessions if x > 0]) / len(sessions)
	print(f"Mean performance from {session_start} to {session_end} on {symbol}: {format_percentage(mean_returns)}")
	print(f"Number of sessions: {len(sessions)}")
	print(f"Samples with positive return: {positive_rate:.2%}")