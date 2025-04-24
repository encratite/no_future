from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from colorama import Fore, Style
from matplotlib.ticker import FuncFormatter

from asset import Asset
from backtest_configuration import BacktestConfiguration
from backtest_interface import BacktestInterface
from backtest_result import BacktestResult
from common import format_percentage, format_money, format_ratio, print_table, format_coord
from configuration import Configuration
from manager import AssetManager
from ohlc import OhlcRecord
from position import Position, PositionSide
from strategy import Strategy

class Backtest:
	_strategies: list[Strategy]
	_configuration: BacktestConfiguration
	_asset_manager: AssetManager
	_time_series: list[pd.Timestamp]
	_time: pd.Timestamp | None
	_cash: float
	_fees: float
	_positions: list[Position]
	_equity_curve: list[float]
	_account_value: float
	_max_account_value: float
	_max_drawdown: float
	_drawdown: list[float]
	_terminated: bool

	def __init__(
		self,
		strategies: list[Strategy],
		configuration: BacktestConfiguration,
		asset_manager: AssetManager
	) -> None:
		self._strategies = strategies
		self._configuration = configuration
		self._asset_manager = asset_manager
		es_series = asset_manager.get_series("ES")
		self._time_series = [x for x in es_series if configuration.start <= x < configuration.end]
		self._time = None
		self._cash = configuration.initial_cash
		self._fees = 0
		self._positions = []
		self._equity_curve = []
		self._account_value = self._cash
		self._max_account_value = self._cash
		self._max_drawdown = 0
		self._drawdown = []
		self._terminated = False

	def run(self) -> BacktestResult:
		assert self._terminated == False
		for strategy in self._strategies:
			strategy.reset()
		trading_days = set(self._time_series)
		for time in self._time_series:
			self._time = time
			signals: defaultdict[str, float] = defaultdict(float)
			interface = BacktestInterface(
				time,
				self._configuration.start,
				self._configuration.end,
				trading_days,
				self._asset_manager
			)
			for strategy in self._strategies:
				strategy_signals = strategy.get_signals(interface)
				for symbol, signal in strategy_signals.items():
					signals[symbol] += strategy.weight * signal
			for symbol in list(signals.keys()):
				if abs(signals[symbol]) < 0.5:
					del signals[symbol]
			self._rebalance(signals)
			self._update_equity_curve()
			self._ruin_check()
			if self._terminated:
				break
		self._close_all_positions()
		result = self._get_result()
		return result

	def plot_equity_curve(self) -> None:
		id_var = "date"
		value_var = "value"
		equity_var = "Equity Curve"
		drawdown_var = "Drawdown"
		value_name = "value_name"
		truncated_time_series = self._time_series[0:len(self._equity_curve)]
		df = pd.DataFrame({
			id_var: truncated_time_series,
			equity_var: self._equity_curve,
			drawdown_var: self._drawdown
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
	def print_result(result: BacktestResult) -> None:
		table = [
			["Net Profit", format_money(result.net_profit)],
			["Annual Average Profit", format_money(result.annual_average_profit)],
			["Starting Capital", format_money(result.starting_capital)],
			["Total Return", format_percentage(result.total_return)],
			["Compound Annual Growth Rate", format_percentage(result.compound_annual_growth_rate)],
			["Sharpe Ratio", format_ratio(result.sharpe_ratio)],
			["Sortino Ratio", format_ratio(result.sortino_ratio)],
			["Max Drawdown", format_percentage(result.max_drawdown)]
		]
		print_table(table, False)

	def _get_result(self) -> BacktestResult:
		result = BacktestResult(
			self._configuration.start,
			self._configuration.end,
			self._equity_curve,
			self._max_drawdown,
			self._configuration.initial_cash,
			self._cash,
			self._asset_manager
		)
		return result

	def _rebalance(self, signals: defaultdict[str, float]) -> None:
		for position in self._positions:
			symbol = position.symbol
			if symbol not in signals:
				self._close_position(symbol)
		for symbol, signal in signals.items():
			signal_count = round(abs(signal))
			signal_side = PositionSide.LONG if signal >= 0 else PositionSide.SHORT
			position_count, position_side = self._get_position_info(symbol)
			if position_side is not None and position_side != signal_side:
				self._close_position(symbol)
				position_count = 0
			delta_count = signal_count - position_count
			if delta_count > 0:
				self._open_position(symbol, delta_count, signal_side)
			elif delta_count < 0:
				count = - delta_count
				self._close_position(symbol, count)

	def _update_equity_curve(self) -> None:
		account_value = self._get_account_value()
		self._max_account_value = max(account_value, self._max_account_value)
		drawdown = account_value - self._max_account_value
		drawdown_percent = account_value / self._max_account_value - 1
		if drawdown_percent < self._max_drawdown:
			self._max_drawdown = drawdown_percent
		self._account_value = account_value
		self._equity_curve.append(account_value)
		self._drawdown.append(drawdown)

	def _ruin_check(self) -> None:
		ratio = self._account_value / self._configuration.initial_cash
		if ratio < Configuration.RUIN_RATIO:
			if self._configuration.enable_output:
				print(f"Account value dropped to {Fore.RED}{ratio:.2%}{Style.RESET_ALL} at {self._time}, terminating simulation prematurely")
				print("Strategies in use by backtest:")
				for i, strategy in enumerate(self._strategies):
					print(f"{i + 1}. {strategy.name}")
			self._terminated = True

	@staticmethod
	def _get_mean_spread(contracts: int, asset: Asset) -> float:
		base_spread_contracts = min(contracts, asset.depth)
		ticks_sum = asset.spread * base_spread_contracts
		remaining_contracts = contracts - base_spread_contracts
		for i in range(remaining_contracts):
			ticks_sum += asset.spread + i + 1
		mean_spread = ticks_sum * asset.tick_value / contracts
		return mean_spread

	def _open_position(self, symbol: str, count: int, side: PositionSide) -> None:
		assert count > 0
		current_record = self._asset_manager.get_record(symbol, self._time)
		asset = self._asset_manager.get_asset(symbol)
		maintenance_margin = self._get_margin(current_record, asset)
		maintenance_margin, forex_fee = self._convert_currency(maintenance_margin, asset.currency)
		initial_margin = count * Configuration.INITIAL_MARGIN_RATIO * maintenance_margin
		if Configuration.ENABLE_FEES:
			fees = forex_fee + asset.broker_fee + asset.exchange_fee
		else:
			fees = 0
		if initial_margin + fees >= self._cash:
			if self._configuration.enable_output:
				print(f"Not enough cash to open a position with {count} contract(s) of {symbol} with an initial margin requirement of {format_money(initial_margin)}")
			return
		cost = count * maintenance_margin + fees
		self._cash -= cost
		self._fees += fees
		if Configuration.ENABLE_FEES:
			mean_spread = self._get_mean_spread(count, asset)
		else:
			mean_spread = 0
		ask = current_record.close + mean_spread * asset.tick_size
		position = Position(
			symbol,
			asset,
			count,
			side,
			ask,
			maintenance_margin,
			self._time
		)
		self._positions.append(position)

	def _close_position(self, symbol: str, count: int | None = None) -> None:
		assert count is None or count > 0
		if count is None:
			matching_positions = [x for x in self._positions if x.symbol == symbol]
			for position in matching_positions:
				value, _bid, fees = self._get_position_value(position, position.count)
				self._cash += value
				self._fees += fees
				self._positions.remove(position)
		else:
			while count > 0:
				position = next((x for x in self._positions if x.symbol == symbol), None)
				assert position is not None
				close_count = min(position.count, count)
				value, _bid, fees = self._get_position_value(position, close_count)
				self._cash += value
				self._fees += fees
				new_count = position.count - close_count
				count -= close_count
				if new_count > 0:
					position.count = new_count
				else:
					self._positions.remove(position)

	def _close_all_positions(self) -> None:
		for position in list(self._positions):
			self._close_position(position.symbol, position.count)

	def _get_position_info(self, symbol: str) -> tuple[int, PositionSide | None]:
		matching_positions = [x for x in self._positions if x.symbol == symbol]
		counts = [x.count for x in matching_positions]
		count = sum(counts)
		if len(matching_positions) > 0:
			side = matching_positions[0].side
		else:
			side = None
		return count, side

	def _get_position_value(self, position: Position, count: int) -> tuple[float, float, float]:
		assert 1 <= count <= position.count
		record = self._asset_manager.get_record(position.symbol, self._time)
		margin = count * position.margin
		bid = record.close
		asset = position.asset
		ticks = count * (bid - position.price) / asset.tick_size
		profit = ticks * asset.tick_value
		if position.side == PositionSide.SHORT:
			profit = -profit
		gain, forex_fee = self._convert_currency(profit, asset.currency)
		if Configuration.ENABLE_FEES:
			fees = forex_fee + asset.broker_fee + asset.exchange_fee
		else:
			fees = 0
		value = margin + gain - fees
		return value, bid, fees

	def _get_account_value(self) -> float:
		account_value = self._cash
		for position in self._positions:
			position_value, _, _ = self._get_position_value(position, position.count)
			account_value += position_value
		return account_value

	@staticmethod
	def _get_margin(current_record: OhlcRecord, asset: Asset) -> float:
		margin = current_record.close / asset.margin_close * asset.margin
		return margin

	def _convert_currency(self, amount: float, currency: str) -> tuple[float, float]:
		if currency == "USD":
			return amount, 0
		exchange_rate = self._asset_manager.get_currency(currency, self._time)
		exchanged_amount = exchange_rate * amount / Configuration.FOREX_SPREAD
		return exchanged_amount, Configuration.FOREX_ORDER_FEE