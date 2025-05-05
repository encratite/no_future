from math import sqrt
from statistics import mean, stdev
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from common import (
	format_percentage,
	format_money,
	format_ratio,
	print_table,
	format_coord
)
from constant import DAYS_PER_YEAR, TRADING_DAYS_PER_YEAR
from manager import AssetManager

class BacktestResult:
	time_series: list[pd.Timestamp]
	equity_curve: list[float]
	drawdown: list[float]
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
		time_series: list[pd.Timestamp],
		equity_curve: list[float],
		drawdown: list[float],
		max_drawdown: float,
		initial_cash: float,
		final_cash: float,
		trades: int,
		profitable_trades: int,
		asset_manager: AssetManager
	):
		self.time_series = time_series
		self.equity_curve = equity_curve
		self.drawdown = drawdown
		self.net_profit = final_cash - initial_cash
		years = (end - start) / pd.Timedelta(days=DAYS_PER_YEAR)
		self.annual_average_profit = self.net_profit / years
		self.starting_capital = initial_cash
		return_ratio = final_cash / initial_cash
		self.total_return = return_ratio - 1
		self.mean_annual_return = self.total_return / years
		self.trades = trades
		if trades > 0:
			self.hit_rate = profitable_trades / trades
		else:
			self.hit_rate = 0
		self.max_drawdown = max_drawdown
		risk_free_rate = self._get_risk_free_rate(start, end, asset_manager)
		sharpe_ratio, sortino_ratio = self._get_ratios(equity_curve, risk_free_rate)
		self.sharpe_ratio = sharpe_ratio
		self.sortino_ratio = sortino_ratio

	def print(self) -> None:
		table = [
			["Net Profit", format_money(self.net_profit)],
			["Annual Average Profit", format_money(self.annual_average_profit)],
			["Starting Capital", format_money(self.starting_capital)],
			["Total Return", format_percentage(self.total_return)],
			["Mean Annual Return", format_percentage(self.mean_annual_return)],
			["Sharpe Ratio", format_ratio(self.sharpe_ratio)],
			["Sortino Ratio", format_ratio(self.sortino_ratio)],
			["Max Drawdown", format_percentage(self.max_drawdown)],
			["Round-Trips", self.trades],
			["Hit Rate", f"{self.hit_rate:.1%}"]
		]
		print_table(table, False)

	def plot(self) -> None:
		id_var = "date"
		value_var = "value"
		equity_var = "Equity Curve"
		drawdown_var = "Drawdown"
		value_name = "value_name"
		truncated_time_series = self.time_series[0:len(self.equity_curve)]
		df = pd.DataFrame({
			id_var: truncated_time_series,
			equity_var: self.equity_curve,
			drawdown_var: self.drawdown
		})
		df_melted = df.melt(
			id_vars=id_var,
			value_vars=[equity_var, drawdown_var],
			var_name=value_var,
			value_name=value_name
		)
		fig, ax = plt.subplots(figsize=(12, 8))
		sns.lineplot(df_melted, x=id_var, y=value_name, hue=value_var)
		ax.legend().set_title(None) # type: ignore
		fill_alpha = 0.1
		ax.fill_between(
			df[id_var],
			df[equity_var],
			0,
			where=(df[equity_var] >= 0),
			interpolate=True,
			color="blue",
			alpha=fill_alpha
		)
		ax.fill_between(
			df[id_var],
			0,
			df[drawdown_var],
			where=(df[drawdown_var] < 0),
			interpolate=True,
			color="red",
			alpha=fill_alpha
		)
		plt.xlim(df[id_var].min(), df[id_var].max())
		plt.xlabel("Date")
		plt.ylabel("Capital")
		plt.title(f"Equity Curve")
		plt.tight_layout()

		def format_money_plot(x, _pos):
			if x >= 0:
				return f"${x:,.2f}"
			else:
				return f"-${abs(x):,.2f}"

		formatter = FuncFormatter(format_money_plot)
		plt.gca().yaxis.set_major_formatter(formatter)
		ax.format_coord = lambda x, y: format_coord(x, y, ax, format_string=lambda x: format_money(x, False))
		plt.show()
		plt.close()

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