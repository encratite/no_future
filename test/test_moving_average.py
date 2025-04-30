from time import perf_counter

import pandas as pd

from backtest.backtest import Backtest, BacktestConfiguration
from manager import AssetManager
from strategy import *

def perform_backtest(start: pd.Timestamp, end: pd.Timestamp) -> None:
	cash = 100_000
	start_time = perf_counter()
	symbol = "ES"
	asset_manager = AssetManager([symbol])
	end_time = perf_counter()
	delta = end_time - start_time
	print(f"Loaded assets in {delta:.1f} s")
	configuration = MovingAverageConfiguration(
		fast_days=8,
		slow_days=20,
		trading_mode=MovingAverageTradingMode.LONG,
		function=MovingAverageFunction.EXPONENTIAL,
		regime_filter=False,
		holding_time=3
	)
	moving_average_strategy = MovingAverageStrategy(symbol, configuration)
	configuration = BacktestConfiguration(start, end, cash)
	backtest = Backtest([moving_average_strategy], configuration, asset_manager)
	result = backtest.run()
	result.print()
	end_time = perf_counter()
	delta = end_time - start_time
	print(f"Performed backtest in {delta:.1f} s")
	# result.plot()