from collections import defaultdict

import pandas as pd

from asset import Asset
from assets import get_asset, ASSET_MARGIN_DATE
from backtest_configuration import BacktestConfiguration
from backtest_interface import BacktestInterface
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

	def _initialize_assets(self) -> None:


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
		raise NotImplementedError()

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

	def _get_margin(self, current_record: OhlcRecord, asset: Asset) -> float:
		margin = current_record.close / asset.margin_close * asset.margin
		return margin