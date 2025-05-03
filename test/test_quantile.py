from time import perf_counter

import pandas as pd

from backtest.backtest import Backtest, BacktestConfiguration
from manager import AssetManager
from strategy import *

def perform_backtest(start: pd.Timestamp, end: pd.Timestamp) -> None:
	cash = 100_000
	start_time = perf_counter()
	symbol = "GC"
	asset_manager = AssetManager([symbol])
	end_time = perf_counter()
	delta = end_time - start_time
	print(f"Loaded assets in {delta:.1f} s")
	quantile_radius_strategy = QuantileRadiusStrategy(symbol, QuantileFeatures.MOMENTUM2, QuantileFeatures.VOLUME, 0.25)
	quantile_radius_strategy.weight = 1
	configuration = BacktestConfiguration(start, end, cash)
	backtest = Backtest([quantile_radius_strategy], configuration, asset_manager)
	result = backtest.run()
	result.print()
	end_time = perf_counter()
	delta = end_time - start_time
	print(f"Performed backtest in {delta:.1f} s")
	result.plot()