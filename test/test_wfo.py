import os
from collections import defaultdict
from itertools import product, chain
from multiprocessing import Pool
from time import perf_counter

import pandas as pd

from backtest.backtest import (
	Backtest,
	BacktestConfiguration,
	BacktestResult
)
from manager import AssetManager
from strategy import *

def perform_wfo_backtest(symbol: str, start: pd.Timestamp, end: pd.Timestamp, wfo_years: int) -> None:
	initial_cash = 100_000
	use_multiprocessing = True

	time_start = perf_counter()

	wfo_dates = []
	previous_date: pd.Timestamp | None = None
	time = start
	while time < end:
		if previous_date is None or (time.month != previous_date.month and time.month in [1, 6]):
			wfo_dates.append(time)
		previous_date = time
		time += pd.Timedelta(days=1)
	asset_manager = AssetManager([symbol])
	cpu_count = os.cpu_count()
	if use_multiprocessing:
		pool_arguments = []
		for process_id in range(cpu_count):
			pool_arguments.append((symbol, wfo_dates, wfo_years, asset_manager, initial_cash, process_id))
		with Pool(cpu_count) as pool:
			output = pool.starmap(get_best_wfo_parameters, pool_arguments)
		wfo_strategies: list[tuple[pd.Timestamp, Strategy]] = list(chain.from_iterable(output))
	else:
		wfo_strategies = get_best_wfo_parameters(symbol, wfo_dates, wfo_years, asset_manager, initial_cash, None)
	wfo_strategy = WfoStrategy(wfo_strategies)
	backtest_configuration = BacktestConfiguration(start, end, initial_cash, False)
	perform_buy_and_hold_test(symbol, backtest_configuration, asset_manager)
	backtest = Backtest([wfo_strategy], backtest_configuration, asset_manager)
	result = backtest.run()
	print_wfo_strategies(wfo_years, wfo_strategies)
	print("WFO performance:")
	backtest.print_result(result)
	backtest.plot_equity_curve()

	time_end = perf_counter()
	time_delta = time_end - time_start
	print(f"Finished WFO backtest in {time_delta:.1f} s")

def perform_buy_and_hold_test(symbol: str, configuration: BacktestConfiguration, asset_manager: AssetManager) -> None:
	results: list[tuple[BacktestResult, str]] = []
	for signal, side_string in [(1, "long"), (-1, "short")]:
		buy_and_hold_signals = {
			symbol: signal
		}
		strategies = [
			BuyAndHoldStrategy(buy_and_hold_signals)
		]
		backtest = Backtest(strategies, configuration, asset_manager)
		result = backtest.run()
		results.append((result, side_string))
	best_result, side_string = max(results, key=lambda x: x[0].sharpe_ratio)
	print(f"Buy and hold performance ({side_string}):")
	backtest.print_result(best_result)

def print_wfo_strategies(wfo_years: int, wfo_strategies: list[tuple[pd.Timestamp, Strategy]]) -> None:
	strategies_dict: defaultdict[str, int] = defaultdict(int)
	for _, strategy in wfo_strategies:
		strategies_dict[strategy.name] += 1
	strategies = list(strategies_dict.items())
	strategies = sorted(strategies, key=lambda x: x[1], reverse=True)
	i = 1
	print(f"Parameters selected by a WFO window of {wfo_years} year(s):")
	for strategy, frequency in strategies:
		print(f"\t{i}. {strategy}: {frequency}")
		i += 1
	print("")

def get_best_wfo_parameters(
	symbol: str,
	wfo_dates: list[pd.Timestamp],
	wfo_years: int,
	asset_manager: AssetManager,
	initial_cash: int,
	process_id: int | None
) -> list[tuple[pd.Timestamp, Strategy]]:
	fast_days = [
		4,
		5,
		6,
		8
	]
	slow_days = [
		10,
		12,
		15,
		20
	]
	fast_slow_values: list[tuple[int, int]] = list(product(fast_days, slow_days))
	fast_slow_values += [
		(25, 100),
		(50, 200)
	]
	trading_modes = [
		MovingAverageTradingMode.LONG,
		MovingAverageTradingMode.SHORT,
		MovingAverageTradingMode.LONG_SHORT,
	]
	regime_filters = [
		False,
		# True
	]
	holding_times = [
		1,
		2,
		3,
		5,
	]
	strategy_parameters = list(product(fast_slow_values, trading_modes, list(MovingAverageFunction), regime_filters, holding_times))
	output: list[tuple[pd.Timestamp, Strategy]] = []
	i = 0
	cpu_count = os.cpu_count()
	for wfo_date in wfo_dates:
		if process_id is not None and i % cpu_count != process_id:
			i += 1
			continue
		start = wfo_date - pd.DateOffset(years=wfo_years)
		end = wfo_date
		backtest_configuration = BacktestConfiguration(start, end, initial_cash)
		backtest_results: list[tuple[Strategy, BacktestResult]] = []
		for fast_slow_tuple, trading_mode, function, regime_filter, holding_time in strategy_parameters:
			fast_days, slow_days = fast_slow_tuple
			strategy_configuration = MovingAverageConfiguration(
				fast_days=fast_days,
				slow_days=slow_days,
				trading_mode=trading_mode,
				function=function,
				regime_filter=regime_filter,
				holding_time=holding_time
			)
			moving_average_strategy = MovingAverageStrategy(symbol, strategy_configuration)
			backtest = Backtest([moving_average_strategy], backtest_configuration, asset_manager)
			result = backtest.run()
			backtest_results.append((moving_average_strategy, result))
		best_strategy, _ = max(backtest_results, key=lambda x: convert_sharpe(x[1].sharpe_ratio))
		output.append((wfo_date, best_strategy))
		i += 1
	return output

def convert_sharpe(sharpe_ratio: float | None) -> float:
	if sharpe_ratio is not None:
		return sharpe_ratio
	else:
		return -10