import os
from math import floor
from typing import cast, Final

import pandas as pd

from configuration import Configuration
from series import TimeSeries, OhlcRecord
from common import get_rate_of_change, print_table

TEST_PERIOD: Final[pd.Timestamp] = pd.Timestamp("2000-01-01")
INITIAL_CASH: Final[float] = 50_000
ONE_MONTH: Final[pd.Timedelta] = pd.Timedelta(days=30)

def compare_dca(symbol: str, dca_amount: float) -> (float, float, int):
	path = os.path.join(Configuration.BARCHART_DIRECTORY, f"{symbol}.D1.csv")
	series = TimeSeries.read_ohlc_csv(path)
	start = TEST_PERIOD
	start_limit = pd.Timestamp("2024-01-01")
	end = pd.Timestamp.now()
	lump_sum_wins = 0
	dca_wins = 0
	while start < start_limit:
		lump_sum_returns = get_lump_sum_returns(start, end, series)
		dca_returns = get_dca_returns(dca_amount, start, end, series)
		if lump_sum_returns > dca_returns:
			lump_sum_wins += 1
		else:
			dca_wins += 1
		start += ONE_MONTH
	runs = lump_sum_wins + dca_wins
	lump_sum_win_ratio = lump_sum_wins / runs
	dca_win_ratio = dca_wins / runs
	return lump_sum_win_ratio, dca_win_ratio, runs

def get_lump_sum_returns(start: pd.Timestamp, end: pd.Timestamp, series: TimeSeries[OhlcRecord]) -> float:
	first_close = cast(OhlcRecord, series.get(start)).close
	last_close = cast(OhlcRecord, series.get(end)).close
	cash = INITIAL_CASH
	shares = floor(cash / first_close)
	cash -= shares * first_close
	cash += shares * last_close
	returns = get_rate_of_change(cash, INITIAL_CASH)
	return returns

def get_dca_returns(dca_amount: float, start: pd.Timestamp, end: pd.Timestamp, series: TimeSeries[OhlcRecord]) -> float:
	cash = INITIAL_CASH
	shares = 0
	now = start
	while now < end:
		close = cast(OhlcRecord, series.get(now)).close
		dca_cash = min(cash, dca_amount)
		dca_shares = floor(dca_cash / close)
		cash -= dca_shares * close
		shares += dca_shares
		now += ONE_MONTH
	last_close = cast(OhlcRecord, series.get(end)).close
	cash += shares * last_close
	returns = get_rate_of_change(cash, INITIAL_CASH)
	return returns

def compare_multiple_dca_amounts(symbol: str):
	dca_amounts = [
		1_000,
		2_000,
		3_000,
		4_000,
		5_000,
		10_000
	]
	print(f"Evaluating different investment methods for {symbol} starting {TEST_PERIOD.strftime("%Y-%m-%d")} with ${INITIAL_CASH:,.2f} to invest:\n")
	headers = [
		"Monthly DCA Amount",
		"Lump Sum Win Ratio",
		"DCA Win Ratio"
	]
	table = [headers]
	runs = None
	for dca_amount in dca_amounts:
		lump_sum_win_ratio, dca_win_ratio, runs = compare_dca(symbol, dca_amount)
		row = [
			f"${dca_amount:,.2f}",
			f"{lump_sum_win_ratio:.1%}",
			f"{dca_win_ratio:.1%}",
		]
		table.append(row)
	print_table(table, always_right=True)
	print(f"Different starting points evaluated per run, with increments of approximately one month: {runs}\n")

compare_multiple_dca_amounts("SPY")