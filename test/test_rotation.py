from itertools import product
from multiprocessing import Pool, cpu_count

import pandas as pd

from backtest.backtest import (
	Backtest,
	BacktestConfiguration
)
from backtest.result import BacktestResult
from manager import AssetManager
from strategy import *

def perform_backtest(start: pd.Timestamp, end: pd.Timestamp) -> None:
	backtest_results = []
	with Pool() as pool:
		processes = cpu_count()
		arguments = [(start, end, i) for i in range(processes)]
		output = pool.starmap(run_process, arguments)
		for results in output:
			backtest_results += results
	backtest_results = sorted(backtest_results, key=lambda x: x.sharpe_ratio, reverse=True)
	for i, result in enumerate(backtest_results):
		print(f"{i + 1}. {result.strategies[0]}: {result.sharpe_ratio:.2f}")
	best_result: BacktestResult = backtest_results[0]
	best_result.print()
	best_result.plot()

def run_process(start: pd.Timestamp, end: pd.Timestamp, process_id: int) -> list[BacktestResult]:
	cash = 100_000
	symbols = [
		"6A",
		# "6B",
		"6C",
		"6E",
		"6J",
		"6M",
		# "6N",
		"6S",
		# "6Z"
	]
	momentum_days_values = [
		5,
		10,
		20,
		90,
		152,
		252,
	]
	position_values = [
		(0, 1),
		(1, 0),
		(1, 1),
		(2, 0),
		(2, 1),
		(2, 2),
	]
	asset_manager = AssetManager(symbols)
	parameters = product(momentum_days_values, position_values)
	processes = cpu_count()
	backtest_results = []
	for i, parameter_tuple in enumerate(parameters):
		if i % processes != process_id:
			continue
		momentum_days, positions = parameter_tuple
		long_positions, short_positions = positions
		rotation_strategy = RotationStrategy(symbols, momentum_days, long_positions, short_positions)
		configuration = BacktestConfiguration(start, end, cash)
		backtest = Backtest([rotation_strategy], configuration, asset_manager)
		backtest_result = backtest.run()
		backtest_results.append(backtest_result)
	return backtest_results