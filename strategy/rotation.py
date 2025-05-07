import pandas as pd

from .base import Strategy
from backtest.interface import BacktestInterface
from common import get_rate_of_change

class RotationStrategy(Strategy):
	_symbols: list[str]
	_momentum_days: int
	_long_positions: int
	_short_positions: int

	_last_rebalance: pd.Timestamp | None
	_previous_signals: dict[str, float] | None

	def __init__(
		self,
		symbols: list[str],
		momentum_days: int,
		long_positions: int,
		short_positions: int
	) -> None:
		super().__init__(f"Rotation ({momentum_days} day window, {long_positions} long, {short_positions} short)")
		self._symbols = symbols
		self._momentum_days = momentum_days
		self._long_positions = long_positions
		self._short_positions = short_positions
		self.reset()

	def get_signals(self, interface: BacktestInterface) -> dict[str, float]:
		if self._last_rebalance is None or interface.time.week != self._last_rebalance.week:
			momentum_values: list[tuple[str, float]] = []
			for symbol in self._symbols:
				records = interface.get_records(symbol, count=self._momentum_days)
				momentum = get_rate_of_change(records[0].close, records[-1].close)
				momentum_values.append((symbol, momentum))
			momentum_values = sorted(momentum_values, key=lambda x: x[1], reverse=True)
			momentum_symbols = [x[0] for x in momentum_values]
			long_symbols = momentum_symbols[:self._long_positions]
			short_symbols = momentum_symbols[-self._short_positions:]
			signals = {}
			for long_symbol in long_symbols:
				signals[long_symbol] = 1
			for short_symbol in short_symbols:
				signals[short_symbol] = -1
			self._last_rebalance = interface.time
			self._previous_signals = signals
			return signals
		else:
			return self._previous_signals

	def reset(self) -> None:
		self._last_rebalance = None
		self._previous_signals = None