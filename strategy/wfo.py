import pandas as pd

from backtest.interface import BacktestInterface
from .base import Strategy

class WfoStrategy(Strategy):
	_strategies: list[tuple[pd.Timestamp, Strategy]]
	_remaining_strategies: list[tuple[pd.Timestamp, Strategy]]
	_active_strategy: Strategy

	def __init__(self, strategies: list[tuple[pd.Timestamp, Strategy]]):
		super().__init__("WFO Strategy")
		self._strategies = strategies
		self.reset()

	def get_signals(self, interface: BacktestInterface) -> dict[str, float]:
		if len(self._remaining_strategies) > 0:
			time, strategy = self._remaining_strategies[0]
			if interface.time >= time:
				self._remaining_strategies.pop(0)
		signals = self._active_strategy.get_signals(interface)
		return signals

	def reset(self) -> None:
		self._remaining_strategies = self._strategies[1:]
		_, self._active_strategy = self._strategies[0]