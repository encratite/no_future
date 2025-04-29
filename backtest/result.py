from math import sqrt
from statistics import mean, stdev
from typing import Final

import pandas as pd

from constant import DAYS_PER_YEAR, TRADING_DAYS_PER_YEAR
from manager import AssetManager

class BacktestResult:
	net_profit: float
	annual_average_profit: float
	starting_capital: float
	total_return: float
	mean_annual_return: float
	trades: int
	hit_rate: float
	max_drawdown: float
	sharpe_ratio: float | None
	sortino_ratio: float | None

	def __init__(
		self,
		start: pd.Timestamp,
		end: pd.Timestamp,
		equity_curve: list[float],
		max_drawdown: float,
		initial_cash: float,
		final_cash: float,
		trades: int,
		profitable_trades: int,
		asset_manager: AssetManager
	):
		self.net_profit = final_cash - initial_cash
		years = (end - start) / pd.Timedelta(days=DAYS_PER_YEAR)
		self.annual_average_profit = self.net_profit / years
		self.starting_capital = initial_cash
		return_ratio = final_cash / initial_cash
		self.total_return = return_ratio - 1
		self.mean_annual_return = self.total_return / years
		self.trades = trades
		self.hit_rate = profitable_trades / trades
		self.max_drawdown = max_drawdown
		risk_free_rate = self._get_risk_free_rate(start, end, asset_manager)
		sharpe_ratio, sortino_ratio = self._get_ratios(equity_curve, risk_free_rate)
		self.sharpe_ratio = sharpe_ratio
		self.sortino_ratio = sortino_ratio

	@staticmethod
	def _get_risk_free_rate(
		start: pd.Timestamp,
		end: pd.Timestamp,
		asset_manager: AssetManager
	) -> float:
		time = start
		rates = []
		while time < end:
			rate = asset_manager.get_risk_free_rate(time)
			rates.append(rate)
			time += pd.Timedelta(days=1)
		mean_rate = mean(rates)
		return mean_rate

	@staticmethod
	def _get_ratios(equity_curve: list[float], risk_free_rate: float) -> tuple[float | None, float | None]:
		trading_days_per_year: Final[int] = TRADING_DAYS_PER_YEAR

		daily_returns = [today / yesterday - 1 for today, yesterday in zip(equity_curve[1:], equity_curve)]
		if len(daily_returns) < 2:
			return None, None
		mean_daily_returns = mean(daily_returns)
		daily_standard_deviation = stdev(daily_returns)
		mean_annual_returns = trading_days_per_year * mean_daily_returns
		standard_deviation_factor = sqrt(trading_days_per_year)
		standard_deviation = standard_deviation_factor * daily_standard_deviation
		excess_returns = mean_annual_returns - risk_free_rate
		if standard_deviation == 0:
			return None, None

		sharpe_ratio = excess_returns / standard_deviation
		downside_daily_returns = [x for x in daily_returns if x < 0]
		if len(downside_daily_returns) >= 2:
			daily_downside_standard_deviation = stdev(downside_daily_returns)
			downside_standard_deviation = standard_deviation_factor * daily_downside_standard_deviation
			sortino_ratio = excess_returns / downside_standard_deviation
		else:
			sortino_ratio = None
		return sharpe_ratio, sortino_ratio