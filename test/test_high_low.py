from time import perf_counter
from itertools import product
from multiprocessing import Pool, cpu_count

from colorama import Fore, Style
import pandas as pd

from backtest.backtest import (
	Backtest,
	BacktestConfiguration,
	BacktestResult
)
from common import try_parse_int
from manager import AssetManager
from strategy import *

class MultiBacktestResult:
	strategy: str
	primary_result: BacktestResult
	secondary_result: BacktestResult

	def __init__(
		self,
		strategy: str,
		primary_result: BacktestResult,
		secondary_result: BacktestResult
	) -> None:
		self.strategy = strategy
		self.primary_result = primary_result
		self.secondary_result = secondary_result

def perform_backtest(start: pd.Timestamp, end: pd.Timestamp) -> None:
	symbol = "NG.F2"
	result_count = 30
	start_time = perf_counter()
	asset_manager = AssetManager([symbol])
	pool_arguments = []
	process_count = cpu_count()
	for process_id in range(process_count):
		pool_arguments.append((symbol, start, end, asset_manager, process_id))
	results: list[MultiBacktestResult] = []
	with Pool(process_count) as pool:
		for output in pool.starmap(evaluate_parameters, pool_arguments):
			results += output
	sorted_results = sorted(results, key=lambda x: x.primary_result.sharpe_ratio, reverse=True)
	trimmed_results = sorted_results[:result_count]
	for i, result in enumerate(trimmed_results):
		secondary_string = f"{result.secondary_result.sharpe_ratio:.2f}"
		if result.secondary_result.sharpe_ratio > result.primary_result.sharpe_ratio:
			secondary_string = f"{Fore.GREEN}{secondary_string}{Style.RESET_ALL}"
		print(f"{i + 1}. {result.strategy}: {result.primary_result.sharpe_ratio:.2f} ({secondary_string})")
	print("")
	print("Best strategy:")
	best_result = trimmed_results[0]
	best_result.primary_result.print()
	best_result.primary_result.plot()
	end_time = perf_counter()
	delta = end_time - start_time
	print(f"Performed backtests in {delta:.1f} s")
	while True:
		id_string = input("Enter the ID of a backtest to review: ")
		if id_string == "":
			break
		result_id = try_parse_int(id_string)
		if result_id < 0 or result_id >= len(trimmed_results):
			print("Invalid ID")
			continue
		result = trimmed_results[result_id - 1]
		print(f"{result.strategy}:")
		result.primary_result.print()
		print("Recent backtest:")
		result.secondary_result.print()
		result.primary_result.plot()

def get_range(a: int, b: int) -> list[int]:
	return list(range(a, b + 1))

def evaluate_parameters(symbol: str, start: pd.Timestamp, end: pd.Timestamp, asset_manager: AssetManager, process_id: int) -> list[MultiBacktestResult]:
	secondary_start = pd.Timestamp("2023-01-01")
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
	results: list[MultiBacktestResult] = []
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
		primary_result = backtest.run()
		configuration = BacktestConfiguration(secondary_start, end, cash)
		backtest = Backtest([moving_average_strategy], configuration, asset_manager)
		secondary_result = backtest.run()
		multi_result = MultiBacktestResult(moving_average_strategy.name, primary_result, secondary_result)
		results.append(multi_result)
	return results