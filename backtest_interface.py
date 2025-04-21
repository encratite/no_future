import pandas as pd

from manager import AssetManager
from ohlc import OhlcRecord

class BacktestInterface:
	_time: pd.Timestamp | None
	_trading_days: set[pd.Timestamp]
	_asset_manager: AssetManager

	def __init__(self, time: pd.Timestamp, trading_days: set[pd.Timestamp], asset_manager: AssetManager):
		self._time = time
		self._trading_days = trading_days
		self._asset_manager = asset_manager

	def get_records(self, symbol: str, time: pd.Timestamp | None = None, count: int | None = None) -> list[OhlcRecord]:
		if time > self._time:
			raise Exception("Strategies cannot read data from the future")
		if time is None:
			time = self._time
		records = self._asset_manager.get_records(symbol, time, count)
		return records

	@property
	def time(self) -> pd.Timestamp:
		return self._time

	def is_trading_day(self, time: pd.Timestamp) -> bool:
		return time in self._trading_days