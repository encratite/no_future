from itertools import product
from multiprocessing import cpu_count

import pandas as pd

from backtest.backtest import (
	Backtest,
	BacktestConfiguration
)
from manager import AssetManager
from strategy import *
from .common import MultiBacktestResult, run_backtest_pool, review_backtests

def perform_backtest(start: pd.Timestamp, end: pd.Timestamp) -> None:
	symbol = "AW"
	results = run_backtest_pool(symbol, start, end, evaluate_parameters)
	review_backtests(results)

def get_range(a: int, b: int) -> list[int]:
	return list(range(a, b + 1))

def evaluate_parameters(symbol: str, start: pd.Timestamp, end: pd.Timestamp, asset_manager: AssetManager, process_id: int) -> list[MultiBacktestResult]:
	secondary_start = pd.Timestamp("2024-01-01")
	cash = 50_000
	window_sizes = get_range(5, 20)
	holding_times = get_range(1, 10)
	modes = [
		NewHighLowMode.LONG_ON_HIGH,
		NewHighLowMode.LONG_ON_LOW,
		NewHighLowMode.SHORT_ON_HIGH,
		NewHighLowMode.SHORT_ON_LOW,
	]
	volatility_window_size = 20
	volatility_filters = [
		(None, None),
		# (volatility_window_size, 0.008),
		# (volatility_window_size, 0.010),
		# (volatility_window_size, 0.012)
	]
	parameters = product(window_sizes, holding_times, modes, volatility_filters)
	results: list[MultiBacktestResult] = []
	process_count = cpu_count()
	for i, parameter_tuple in enumerate(parameters):
		if i % process_count != process_id:
			continue
		window_size, holding_time, mode, volatility_configuration = parameter_tuple
		volatility_window_size, volatility_filter = volatility_configuration
		new_high_low_strategy = NewHighLowStrategy(
			symbol=symbol,
			window_size=window_size,
			holding_time=holding_time,
			mode=mode,
			volatility_window_size=volatility_window_size,
			volatility_filter=volatility_filter
		)
		new_high_low_strategy.weight = 10

		configuration = BacktestConfiguration(start, end, cash)
		backtest = Backtest([new_high_low_strategy], configuration, asset_manager)
		primary_result = backtest.run()
		configuration = BacktestConfiguration(secondary_start, end, cash)
		backtest = Backtest([new_high_low_strategy], configuration, asset_manager)
		secondary_result = backtest.run()
		multi_result = MultiBacktestResult(new_high_low_strategy.name, primary_result, secondary_result)
		results.append(multi_result)
	return results