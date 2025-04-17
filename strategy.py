from abc import ABC, abstractmethod

from backtest_interface import BacktestInterface

class Strategy(ABC):
	weight: float

	def __init__(self):
		self.weight = 1

	@abstractmethod
	def get_signals(self, interface: BacktestInterface) -> dict[str, float]:
		pass