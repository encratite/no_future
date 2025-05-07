from random import random
from statistics import mean

import pandas as pd

from common import (
	get_sharpe_ratio,
	format_money
)
from constant import DAYS_PER_YEAR, TRADING_DAYS_PER_YEAR
from manager import AssetManager

def get_simulated_sharpe_ratio(
	trades: int,
	winning_trades: int,
	profit_factor: float,
	start: pd.Timestamp,
	end: pd.Timestamp
) -> float:
	simulated_returns = []
	while len(simulated_returns) < trades:
		if random() > winning_trades / trades:
			pnl = profit_factor
		else:
			pnl = -1
		simulated_returns.append(pnl)
	years = (end - start) / pd.Timedelta(days=DAYS_PER_YEAR)
	trading_days = round(years * TRADING_DAYS_PER_YEAR)
	padding_days = trading_days - trades
	simulated_returns += padding_days * [0]
	sharpe_ratio = get_sharpe_ratio(simulated_returns)
	return sharpe_ratio

def evaluate_sharpe() -> None:
	symbol = "NQ"
	start = pd.Timestamp("2014-03-03")
	end = pd.Timestamp("2025-03-21")
	print(f"\nComparing strategy to buy and hold on {symbol} from {start.date()} to {end.date()}\n")

	asset_manager = AssetManager([symbol])
	series = asset_manager.get_series(symbol)
	asset = asset_manager.get_asset(symbol)
	records = series.values()
	upticks = 0
	total = 0
	nq_returns = []
	for previous_record, record in zip(records, records[1:]):
		if record.time < start:
			continue
		if record.time >= end:
			break
		if record.close > previous_record.close:
			upticks += 1
		total += 1
		returns = (record.close - previous_record.close) / asset.tick_size * asset.tick_value
		nq_returns.append(returns)
	sharpe_ratio = get_sharpe_ratio(nq_returns)
	total_return = sum(nq_returns)
	print(f"Buy and hold hit rate: {upticks / total:.1%}")
	print(f"Buy and hold total return: {format_money(total_return)}")
	print(f"Buy and hold Sharpe ratio: {sharpe_ratio:.2f}\n")

	trades = 233
	winning_trades = 133
	profit_factor = 1.64
	profit = 42976.39
	sharpe_ratios = [get_simulated_sharpe_ratio(trades, winning_trades, profit_factor, start, end) for _ in range(1000)]
	sharpe_ratio = mean(sharpe_ratios)
	print(f"Strategy hit rate: {winning_trades / trades:.1%}")
	print(f"Strategy total return: {format_money(profit)}")
	print(f"Strategy Sharpe ratio: {sharpe_ratio:.2f}\n")

evaluate_sharpe()