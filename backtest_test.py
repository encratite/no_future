from time import perf_counter

import pandas as pd

from manager import AssetManager
from backtest import Backtest
from backtest_configuration import BacktestConfiguration
from strategy import BuyAndHoldStrategy, MomentumStrategy, RebalanceMode

def perform_backtest(start: pd.Timestamp, end: pd.Timestamp) -> None:
	cash = 5_000_000
	target_notional_value = 10_000
	momentum_symbols = [
		# Metals
		"GC",
		"SI",
		"PL",
		"HG",
		# Energy
		"CL",
		"NG.FY",
		# Agriculture
		"ZS",
		"ZL",
		"ZM",
		"ZW",
		"ZC",
		# Softs
		"CT",
		"SB",
		# Meats
		"HE",
		"LE",
		# Currencies
		"6E",
		"6B",
		"6S",
		# Rates
		"ZB",
		"ZN",
		"ZT"
	]

	start_time = perf_counter()
	asset_manager = AssetManager()
	end_time = perf_counter()
	delta = end_time - start_time
	print(f"Loaded assets in {delta:.1f} s")

	start_time = perf_counter()
	# rebalance_modes = list(RebalanceMode)
	rebalance_modes = [RebalanceMode.END_OF_MONTH]
	for rebalance_mode in rebalance_modes:
		buy_and_hold = BuyAndHoldStrategy({"ES": 2})
		buy_and_hold.weight = len(momentum_symbols)
		momentum_1m = MomentumStrategy(1, rebalance_mode, momentum_symbols, target_notional_value)
		momentum_3m = MomentumStrategy(3, rebalance_mode, momentum_symbols, target_notional_value)
		momentum_12m = MomentumStrategy(12, rebalance_mode, momentum_symbols, target_notional_value)
		strategies = [
			# buy_and_hold,
			momentum_1m,
			momentum_3m,
			momentum_12m,
		]
		configuration = BacktestConfiguration(start, end, cash)
		backtest = Backtest(strategies, configuration, asset_manager)
		result = backtest.run()
		print(rebalance_mode)
		backtest.print_result(result)
		# backtest.plot_equity_curve()
	end_time = perf_counter()
	delta = end_time - start_time
	print(f"Performed backtest in {delta:.1f} s")