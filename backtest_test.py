from time import perf_counter

import pandas as pd

from backtest import Backtest
from backtest_configuration import BacktestConfiguration
from manager import AssetManager
from strategy import BuyAndHoldStrategy, MomentumStrategy, RebalanceMode

def perform_backtest(start: pd.Timestamp, end: pd.Timestamp) -> None:
	cash = 500_000
	target_notional_value = 10_000
	momentum_symbols = [
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
		# "SB",
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
		"ZT"
	]

	start_time = perf_counter()
	asset_manager = AssetManager()
	end_time = perf_counter()
	delta = end_time - start_time
	print(f"Loaded assets in {delta:.1f} s")

	start_time = perf_counter()
	rebalance_mode = RebalanceMode.END_OF_MONTH
	portfolio = {}
	for symbol in momentum_symbols:
		portfolio[symbol] = 1
	strategies = [
		BuyAndHoldStrategy(portfolio)
	]
	momentum_months = [
		1,
		2,
		3,
		4,
		5,
		6,
		7,
		12,
		24
	]
	for months in momentum_months:
		strategy = MomentumStrategy(months, rebalance_mode, momentum_symbols, target_notional_value)
		strategies.append(strategy)
	for strategy in strategies:
		print(f"Rebalance: {rebalance_mode}")
		print(f"Strategy: {strategy.name}")
		configuration = BacktestConfiguration(start, end, cash)
		backtest = Backtest([strategy], configuration, asset_manager)
		result = backtest.run()
		backtest.print_result(result)
		# backtest.plot_equity_curve()
	end_time = perf_counter()
	delta = end_time - start_time
	print(f"Performed backtest in {delta:.1f} s")