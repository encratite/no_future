from time import perf_counter
from itertools import product
from multiprocessing import Pool, cpu_count

import pandas as pd

from backtest.backtest import Backtest, BacktestConfiguration, BacktestResult
from manager import AssetManager
from strategy import *

def perform_backtest(start: pd.Timestamp, end: pd.Timestamp) -> None:
	symbol = "NG"

	start_time = perf_counter()
	asset_manager = AssetManager([symbol])
	end_time = perf_counter()
	delta = end_time - start_time
	print(f"Loaded assets in {delta:.1f} s")

	pool_arguments = []
	process_count = cpu_count()
	for process_id in range(process_count):
		pool_arguments.append((symbol, start, end, asset_manager, process_id))
	results = []
	with Pool(process_count) as pool:
		for output in pool.starmap(evaluate_parameters, pool_arguments):
			results += output

	sorted_results = sorted(results, key=lambda x: x[1].sharpe_ratio, reverse=True)
	trimmed_results = sorted_results[:10]
	for i, name_result in enumerate(trimmed_results):
		name, result = name_result
		print(f"{i + 1}. {name}: {result.sharpe_ratio:.2f}")

	print("")
	print("Best strategy:")
	best_result = trimmed_results[0][1]
	best_result.result()
	# best_result.plot()

	end_time = perf_counter()
	delta = end_time - start_time
	print(f"Performed backtests in {delta:.1f} s")

def get_range(a: int, b: int) -> list[int]:
	return list(range(a, b + 1))

def evaluate_parameters(symbol: str, start: pd.Timestamp, end: pd.Timestamp, asset_manager: AssetManager, process_id: int) -> list[BacktestResult]:
	cash = 100_000
	window_sizes = get_range(5, 20)
	holding_times = get_range(1, 10)
	modes = [
		NewHighLowMode.LONG_ON_HIGH,
		NewHighLowMode.LONG_ON_LOW,
		NewHighLowMode.SHORT_ON_HIGH,
		NewHighLowMode.SHORT_ON_LOW,
	]
	parameters = product(window_sizes, holding_times, modes)
	results = []

	process_count = cpu_count()
	for i, parameter_tuple in enumerate(parameters):
		if i % process_count != process_id:
			continue
		window_size, holding_time, mode = parameter_tuple
		moving_average_strategy = NewHighLowStrategy(
			symbol=symbol,
			window_size=window_size,
			holding_time=holding_time,
			mode=mode
		)
		configuration = BacktestConfiguration(start, end, cash)
		backtest = Backtest([moving_average_strategy], configuration, asset_manager)
		result = backtest.run()
		results.append((moving_average_strategy.name, result))
	return results