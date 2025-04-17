from abc import ABC, abstractmethod

from interface import BacktestInterface

class Strategy(ABC):
	@abstractmethod
	def get_signals(self, interface: BacktestInterface) -> dict[str, float]:
		pass