import datetime as dt
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import (
	read_ohlc_series,
	get_rate_of_change,
	format_percentage
)

ID_VAR: Final[str] = "time"
SESSION: Final[str] = "Session"
OVERNIGHT: Final[str] = "Overnight"
VAR_NAME: Final[str] = "value"
VALUE_NAME: Final[str] = "value_name"

DEBUG_OUTPUT: Final[bool] = False

def analyze_session_returns(symbol: str, session_start: dt.time, session_end: dt.time, start: pd.Timestamp, end: pd.Timestamp) -> None:
	assert session_start < session_end
	series = read_ohlc_series(symbol, intraday=True)
	records = series.values()
	session_opens: list[tuple[pd.Timestamp, float]] = []
	session_closes: list[float] = []
	for i, record in enumerate(records):
		if i == 0:
			continue
		if record.time < start:
			continue
		if record.time >= end:
			break
		previous_record = records[i - 1]
		assert previous_record.time < record.time
		record_time = record.time.time()
		looking_for_close = len(session_closes) < len(session_opens)
		valid_close = len(session_opens) - 1 == len(session_closes)
		if record_time == session_start:
			assert len(session_opens) == len(session_closes)
			session_opens.append((record.time.normalize(), previous_record.close))
			if DEBUG_OUTPUT:
				print(record.time, previous_record.close, "New session")
		elif record_time == session_end and looking_for_close:
			assert valid_close
			session_closes.append(previous_record.close)
			if DEBUG_OUTPUT:
				print(record.time, previous_record.close, f"End of session")
		elif previous_record.time.normalize() != record.time.normalize() and looking_for_close:
			assert valid_close
			session_closes.append(previous_record.close)
			if DEBUG_OUTPUT:
				print(previous_record.time, previous_record.close, f"End of session (weekend)")
	session_times: list[pd.Timestamp] = []
	session_returns: list[float] = []
	overnight_returns: list[float] = []
	for open_tuple1, session_close, open_tuple2 in zip(session_opens, session_closes, session_opens[1:]):
		session_time, session_open = open_tuple1
		_, next_session_open = open_tuple2
		session_return = get_rate_of_change(session_close, session_open)
		if DEBUG_OUTPUT:
			print(f"{session_time} {format_percentage(session_return)} ({session_open} -> {session_close})")
		overnight_return = get_rate_of_change(next_session_open, session_close)
		session_times.append(session_time)
		session_returns.append(session_return)
		overnight_returns.append(overnight_return)
	session_equity_curve = get_equity_curve(session_returns)
	overnight_equity_curve = get_equity_curve(overnight_returns)
	df = pd.DataFrame({
		ID_VAR: session_times,
		SESSION: session_equity_curve,
		OVERNIGHT: overnight_equity_curve
	})
	df_melted = df.melt(
		id_vars=ID_VAR,
		value_vars=[SESSION, OVERNIGHT],
		var_name=VAR_NAME,
		value_name=VALUE_NAME
	)
	_fig, ax = plt.subplots(figsize=(12, 8))
	sns.lineplot(df_melted, ax=ax, x=ID_VAR, y=VALUE_NAME, hue=VAR_NAME)
	x_min = df_melted[ID_VAR].min()
	x_max = df_melted[ID_VAR].max()
	plt.xlim(x_min, x_max)
	plt.xlabel("Time")
	plt.ylabel("Equity")
	plt.title(f"Session vs. Overnight Returns of {symbol}")
	ax.legend().set_title(None) # type: ignore
	plt.tight_layout()
	plt.show()
	plt.close()

def get_equity_curve(returns: list[float]) -> list[float]:
	cash = 1
	equity_curve = []
	for x in returns:
		cash *= 1 + x
		equity_curve.append(cash)
	return equity_curve