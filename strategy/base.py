import re
from abc import ABC, abstractmethod

import pandas as pd

from backtest.interface import BacktestInterface

class Strategy(ABC):
	name: str
	weight: float

	def __init__(self, name: str):
		self.name = name
		self.weight = 1

	@abstractmethod
	def get_signals(self, interface: BacktestInterface) -> dict[str, float]:
		pass

	def reset(self) -> None:
		pass

	@staticmethod
	def is_banned_symbol(symbol: str, time: pd.Timestamp) -> bool:
		covid_start = pd.Timestamp("2020-03-01")
		covid_end = pd.Timestamp("2021-03-01")
		if covid_start <= time < covid_end:
			banned_pattern = re.compile(r"^(CL|NG)(\.F.)?")
			if banned_pattern.match(symbol) is not None:
				return True
		if symbol == "ZS" and time < pd.Timestamp("2010-01-01"):
			return True
		elif symbol == "CT" and time < pd.Timestamp("2010-10-01"):
			return True
		return False