import pandas as pd

from manager import AssetManager
from ohlc import OhlcRecord

class BacktestInterface:
	_time: pd.Timestamp
	_start: pd.Timestamp
	_end: pd.Timestamp
	_trading_days: set[pd.Timestamp]
	_asset_manager: AssetManager

	def __init__(
		self,
		time: pd.Timestamp,
		start: pd.Timestamp,
		end: pd.Timestamp,
		trading_days: set[pd.Timestamp],
		asset_manager: AssetManager
	):
		self._time = time
		self._start = start
		self._end = end
		self._trading_days = trading_days
		self._asset_manager = asset_manager

	def get_record(self, symbol: str, time: pd.Timestamp | None = None) -> OhlcRecord:
		time = self._process_time(time)
		record = self._asset_manager.get_record(symbol, time)
		if record is None:
			raise Exception(f"No record for symbol {symbol}")
		return record

	def get_records(self, symbol: str, time: pd.Timestamp | None = None, count: int | None = None) -> list[OhlcRecord]:
		time = self._process_time(time)
		records = self._asset_manager.get_records(symbol, time, count)
		return records

	@property
	def time(self) -> pd.Timestamp:
		return self._time

	@property
	def start(self) -> pd.Timestamp:
		return self._start

	@property
	def end(self) -> pd.Timestamp:
		return self._end

	def is_trading_day(self, time: pd.Timestamp) -> bool:
		return time in self._trading_days

	def get_contracts(self, symbol: str, record: OhlcRecord, target_notional_value: float) -> float:
		asset = self._asset_manager.get_asset(symbol)
		margin = record.close / asset.margin_close * asset.margin
		fractional_contracts = target_notional_value / margin
		return fractional_contracts

	def _process_time(self, time: pd.Timestamp) -> pd.Timestamp:
		if time is None:
			time = self._time
		if time > self._time:
			raise Exception("Strategies cannot read data from the future")
		return time