from glob import glob
import os
import re

import pandas as pd

from asset import Asset
from assets import get_assets
from configuration import Configuration
from ohlc import OhlcRecord
from series import TimeSeries

class AssetManager:
	_time_series: dict[str, TimeSeries[OhlcRecord]]
	_assets: dict[str, Asset]

	def __init__(self):
		self._time_series = {}
		self._assets = {}
		self._load_time_series()
		self._load_assets()

	def get_record(self, symbol: str, time: pd.Timestamp) -> OhlcRecord:
		symbol = self._translate_symbol(symbol)
		time_series = self._time_series[symbol]
		record = time_series.get(time)
		return record

	def get_records(self, symbol: str, time: pd.Timestamp, count: int | None = None) -> list[OhlcRecord]:
		symbol = self._translate_symbol(symbol)
		time_series = self._time_series[symbol]
		records = time_series.get(time, count=count)
		return records

	def get_series(self, symbol: str) -> TimeSeries[OhlcRecord]:
		symbol = self._translate_symbol(symbol)
		return self._time_series[symbol]

	def get_asset(self, symbol: str) -> Asset:
		pattern = re.compile("^[^.]+")
		match = pattern.match(symbol)
		if match is None:
			raise Exception("Unable to parse symbol")
		truncated_symbol = match[0]
		asset = self._assets[truncated_symbol]
		return asset

	def _load_time_series(self) -> None:
		pattern = os.path.join(Configuration.FEATHER_DIRECTORY, "*.feather")
		paths = glob(pattern)
		for path in paths:
			basename = os.path.basename(path)
			symbol, _extension = os.path.splitext(basename)
			time_series = TimeSeries.read_ohlc_feather(path)
			self._time_series[symbol] = time_series

	def _load_assets(self) -> None:
		assets, margin_date = get_assets()
		for asset in assets:
			self._assets[asset.symbol] = asset
			margin_record = self.get_record(asset.symbol, margin_date)
			asset.margin_close = margin_record.close

	@staticmethod
	def _translate_symbol(symbol: str) -> str:
		pattern = re.compile(r"\.F1$")
		if pattern.match(symbol) is None:
			symbol += ".F1"
		return symbol