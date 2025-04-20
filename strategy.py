from abc import ABC, abstractmethod

from backtest_interface import BacktestInterface

class Strategy(ABC):
	name: str
	weight: float

	def __init__(self, name: str):
		self.name = name
		self.weight = 1

	@abstractmethod
	def get_signals(self, interface: BacktestInterface) -> dict[str, float]:
		pass

class BuyAndHoldStrategy(Strategy):
	_signals: dict[str, float]

	def __init__(self, signals: dict[str, float]):
		super().__init__("Buy and Hold")
		self._signals = signals

	def get_signals(self, interface: BacktestInterface) -> dict[str, float]:
		return self._signals