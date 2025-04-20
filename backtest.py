from collections import defaultdict

import pandas as pd

from asset import Asset
from backtest_configuration import BacktestConfiguration
from backtest_interface import BacktestInterface
from configuration import Configuration
from manager import AssetManager
from position import Position, PositionSide
from ohlc import OhlcRecord
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

	def run(self) -> None:
		for time in self._time_series:
			self._time = time
			signals: defaultdict[str, float] = defaultdict(float)
			interface = BacktestInterface(time, self._asset_manager)
			for strategy in self._strategies:
				strategy_signals = strategy.get_signals(interface)
				for symbol, signal in strategy_signals:
					signals[symbol] += strategy.weight * signal
			self._rebalance(signals)

	def _rebalance(self, signals: defaultdict[str, float]) -> None:
		for position in self._positions:
			symbol = position.symbol
			if symbol not in signals:
				self._close_position(symbol)
		for symbol, signal in signals.items():
			signal_count = round(signal)
			signal_side = PositionSide.LONG if signal >= 0 else PositionSide.SHORT
			position_count, position_side = self._get_position_info(symbol)
			if position_side is not None and position_side != signal_side:
				self._close_position(symbol)
				position_count = 0
			delta_count = signal_count - position_count
			if delta_count > 0:
				self._open_position(symbol, delta_count, signal_side)
			elif delta_count < 0:
				self._close_position(symbol, delta_count)

	def _open_position(self, symbol: str, count: int, side: PositionSide) -> None:
		assert count > 0
		current_record = self._asset_manager.get_record(symbol, self._time)
		asset = self._asset_manager.get_asset(symbol)
		maintenance_margin = self._get_margin(current_record, asset)
		maintenance_margin, forex_fee = self._convert_currency(maintenance_margin, asset.currency)
		initial_margin = Configuration.INITIAL_MARGIN_RATIO * maintenance_margin
		fees = forex_fee + asset.broker_fee + asset.exchange_fee
		if initial_margin + fees >= self._cash:
			raise Exception(f"Not enough cash to open a position with {count} contract(s) of {symbol} with an initial margin requirement of ${initial_margin:.2}")
		cost = count * maintenance_margin + fees
		self._cash -= cost
		self._fees += fees
		ask = current_record.close + asset.spread * asset.tick_size
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
		raise NotImplementedError()

	def _get_position_info(self, symbol: str) -> tuple[int, PositionSide | None]:
		matching_positions = [x for x in self._positions if x.symbol == symbol]
		counts = [x.count for x in matching_positions]
		count = sum(counts)
		if len(matching_positions) > 0:
			side = matching_positions[0].side
		else:
			side = None
		return count, side

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