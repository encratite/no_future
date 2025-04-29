from time import perf_counter

import pandas as pd

from backtest.backtest import Backtest, BacktestConfiguration
from manager import AssetManager
from strategy import *

def perform_backtest(start: pd.Timestamp, end: pd.Timestamp) -> None:
	cash = 100_000
	start_time = perf_counter()
	symbols = [
		"ES",
		# "NQ",
		"GC",
		# "SI",
		"CL",
		# "CL.FY",
		# "NG",
		# "NG.FY",
		"ZS",
		# "ZC",
		# "LE",
		# "HE",
		# "ZB",
		# "ZT",
		# "ZN"
	]
	long_only_symbols = [
		"ES",
		"NQ",
		"GC",
		"SI"
	]
	asset_manager = AssetManager(symbols)
	end_time = perf_counter()
	delta = end_time - start_time
	print(f"Loaded assets in {delta:.1f} s")
	daily_momentum_strategy = DailyMomentumStrategy(symbols, long_only_symbols, weeks=8, long_count=2, short_count=1)
	configuration = BacktestConfiguration(start, end, cash)
	backtest = Backtest([daily_momentum_strategy], configuration, asset_manager)
	result = backtest.run()
	backtest.print_result(result)
	end_time = perf_counter()
	delta = end_time - start_time
	print(f"Performed backtest in {delta:.1f} s")
	backtest.plot_equity_curve()