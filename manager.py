from glob import glob
import os
import re

import pandas as pd

from configuration import Configuration
from ohlc import OhlcRecord
from series import TimeSeries

class AssetManager:
	_time_series: dict[str, TimeSeries[OhlcRecord]]

	def __init__(self):
		self._load_time_series()

	def get_records(self, symbol: str, time: pd.Timestamp, count: int | None = None):
		pattern = re.compile(r"\.F1$")
		if pattern.match(symbol) is None:
			symbol += ".F1"
		time_series = self._time_series[symbol]
		records = time_series.get(time, count=count)
		return records

	def _load_time_series(self):
		pattern = os.path.join(Configuration.FEATHER_DIRECTORY, "*.feather")
		paths = glob(pattern)
		for path in paths:
			basename = os.path.basename(path)
			symbol, _extension = os.path.splitext(basename)
			time_series = TimeSeries.read_ohlc_feather(path)
			self._time_series[symbol] = time_series