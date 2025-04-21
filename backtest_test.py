import pandas as pd

from manager import AssetManager
from backtest import Backtest
from backtest_configuration import BacktestConfiguration
from strategy import BuyAndHoldStrategy

def perform_backtest(start: pd.Timestamp, end: pd.Timestamp) -> None:
	buy_and_hold = BuyAndHoldStrategy({"ES": 1, "GC": 1, "NG": -1})
	strategies = [
		buy_and_hold
	]
	configuration = BacktestConfiguration(start, end, 500_000)
	asset_manager = AssetManager()
	backtest = Backtest(strategies, configuration, asset_manager)
	result = backtest.run()
	backtest.print_result(result)
	backtest.plot_equity_curve()