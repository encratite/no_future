from multiprocessing import Pool, cpu_count
from time import perf_counter
from typing import Callable

import pandas as pd
from colorama import Fore, Style

from backtest.backtest import BacktestResult
from common import try_parse_int
from manager import AssetManager

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

def run_backtest_pool(symbol: str, start: pd.Timestamp, end: pd.Timestamp, evaluate_parameters: Callable) -> list[MultiBacktestResult]:
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
	return trimmed_results

def review_backtests(results: list[MultiBacktestResult]) -> None:
	while True:
		id_string = input("Enter the ID of a backtest to review: ")
		if id_string == "":
			break
		result_id = try_parse_int(id_string)
		if result_id < 1 or result_id > len(results):
			print("Invalid ID")
			continue
		result = results[result_id - 1]
		print(f"{result.strategy}:")
		result.primary_result.print()
		print("Recent backtest:")
		result.secondary_result.print()
		result.primary_result.plot()