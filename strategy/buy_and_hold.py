from backtest.interface import BacktestInterface
from .base import Strategy

class BuyAndHoldStrategy(Strategy):
	_signals: dict[str, float]

	def __init__(self, signals: dict[str, float]):
		super().__init__("Buy and Hold")
		self._signals = signals

	def get_signals(self, interface: BacktestInterface) -> dict[str, float]:
		return self._signals