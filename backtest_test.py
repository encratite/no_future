from time import perf_counter
from itertools import product

import pandas as pd

from backtest import Backtest
from backtest_configuration import BacktestConfiguration
from manager import AssetManager
from strategy import DailySeasonalityStrategy

def perform_backtest(start: pd.Timestamp, end: pd.Timestamp) -> None:
	cash = 500_000
	target_notional_value = 10_000
	symbols = [
		# Metals
		# "GC",
		# "SI",
		# "PL",
		# "HG",
		# Energy
		# "CL",
		# "NG.FY",
		# Agriculture
		# "ZS",
		# "ZL",
		# "ZW",
		# "ZC",
		# Softs
		# "CT",
		"SB",
		# Meats
		# "HE",
		# "LE",
		# Currencies
		# "6E",
		# "6B",
		# "6S",
		# Rates
		# "ZB",
		# "ZN",
		# "ZT"
	]

	start_time = perf_counter()
	asset_manager = AssetManager()
	end_time = perf_counter()
	delta = end_time - start_time
	print(f"Loaded assets in {delta:.1f} s")
	sample_size_values = [
		# 250,
		# 500,
		# 750,
		1000
	]
	minimum_values = [
		# 0,
		# 0.0005,
		# 0.0010,
		# 0.0015,
		0.0020,
	]
	strategies = []
	for sample_size, minimum in product(sample_size_values, minimum_values):
		strategy = DailySeasonalityStrategy(sample_size, minimum, symbols, target_notional_value)
		strategies.append(strategy)
	for strategy in strategies:
		print(f"Strategy: {strategy.name}")
		configuration = BacktestConfiguration(start, end, cash)
		backtest = Backtest([strategy], configuration, asset_manager)
		result = backtest.run()
		backtest.print_result(result)
		backtest.plot_equity_curve()
	end_time = perf_counter()
	delta = end_time - start_time
	print(f"Performed backtest in {delta:.1f} s")