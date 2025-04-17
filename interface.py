import pandas as pd

from manager import AssetManager
from ohlc import OhlcRecord

class BacktestInterface:
	_asset_manager: AssetManager
	_time: pd.Timestamp | None

	def __init__(self, asset_manager: AssetManager):
		self._asset_manager = asset_manager
		self._time = None

	def get_time(self) -> pd.Timestamp:
		return self._time

	def set_time(self, time: pd.Timestamp):
		assert self._time is None or time > self._time
		self._time = time

	def get_records(self, symbol: str, time: pd.Timestamp | None = None, count: int | None = None) -> list[OhlcRecord]:
		if time > self._time:
			raise Exception("Strategies cannot read data from the future")
		if time is None:
			time = self._time
		records = self._asset_manager.get_records(symbol, time, count)
		return records