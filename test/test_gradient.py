from itertools import product
from multiprocessing import cpu_count

import pandas as pd

from backtest.backtest import (
	Backtest,
	BacktestConfiguration
)
from manager import AssetManager
from strategy import *
from .common import MultiBacktestResult, run_backtest_pool

def perform_backtest(start: pd.Timestamp, end: pd.Timestamp) -> None:
	symbol = "ZN"
	run_backtest_pool(symbol, start, end, evaluate_parameters)

def evaluate_parameters(symbol: str, start: pd.Timestamp, end: pd.Timestamp, asset_manager: AssetManager, process_id: int) -> list[MultiBacktestResult]:
	secondary_start = pd.Timestamp("2023-01-01")
	cash = 50_000
	window_sizes = [
		20,
		30,
		40,
		50,
		75,
		100
	]
	signal_counts = [
		1,
		2,
		3,
		4,
	]
	regime_filter_values = [
		False,
		True
	]
	invert_values = [
		False,
		True
	]
	parameters = product(window_sizes, signal_counts, regime_filter_values, invert_values)
	results: list[MultiBacktestResult] = []
	process_count = cpu_count()
	for i, parameter_tuple in enumerate(parameters):
		if i % process_count != process_id:
			continue
		window_size, signal_count, regime_filter, invert = parameter_tuple
		moving_average_strategy = GradientStrategy(
			symbol=symbol,
			window_size=window_size,
			signal_count=signal_count,
			regime_filter=regime_filter,
			invert=invert
		)
		configuration = BacktestConfiguration(start, end, cash)
		backtest = Backtest([moving_average_strategy], configuration, asset_manager)
		primary_result = backtest.run()
		configuration = BacktestConfiguration(secondary_start, end, cash)
		backtest = Backtest([moving_average_strategy], configuration, asset_manager)
		secondary_result = backtest.run()
		multi_result = MultiBacktestResult(moving_average_strategy.name, primary_result, secondary_result)
		results.append(multi_result)
	return results