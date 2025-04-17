from typing import Final
from enum import Enum

import pandas as pd

from asset import Asset

class PositionSide(Enum):
	LONG: Final[int] = 1
	SHORT: Final[int] = 0

class Position:
	symbol: str
	asset: Asset
	count: int
	side: PositionSide
	price: float
	margin: float
	time_opened: pd.Timestamp

	def __init__(
		self,
		symbol: str,
		asset: Asset,
		count: int,
		side: PositionSide,
		price: float,
		margin: float,
		time_opened: pd.Timestamp
	):
		self.symbol = symbol
		self.asset = asset
		self.count = count
		self.side = side
		self.price = price
		self.margin = margin
		self.time_opened = time_opened