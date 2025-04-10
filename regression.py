import numpy as np
import numpy.typing as npt
import pandas as pd

from common import read_ohlc_series, get_rate_of_change, execute_thread_pool
from ohlc import OhlcRecord
from series import TimeSeries

def analyze_momentum(symbols: list[str], start: pd.Timestamp, split: pd.Timestamp, end: pd.Timestamp) -> None:
	pass

def analyze_momentum_by_symbol(symbol: str, start: pd.Timestamp, split: pd.Timestamp, end: pd.Timestamp) -> None:
	assert start < split < end
	ohlc_series = read_ohlc_series(symbol)
	training_times = [x for x in ohlc_series if start <= x < split]
	validation_times = [x for x in ohlc_series if split <= x < end]
	training_samples = len(training_times)
	validation_samples = len(validation_times)
	sequential_returns = 20
	feature_count = sequential_returns
	x_training = np.empty((training_samples, feature_count), dtype=np.float64)
	y_training = np.empty(training_samples, dtype=np.float64)
	x_validation = np.empty((validation_samples, feature_count), dtype=np.float64)
	y_validation = np.empty(validation_samples, dtype=np.float64)
	get_sequential_returns(sequential_returns, ohlc_series, training_times, x_training, y_training)
	get_sequential_returns(sequential_returns, ohlc_series, validation_times, x_validation, y_validation)

def get_sequential_returns(sequential_returns: int, ohlc_series: TimeSeries[OhlcRecord], times: list[pd.Timestamp], x: npt.NDArray, y: npt.NDArray) -> None:
	for i, time in enumerate(times):
		records = ohlc_series.get(time + pd.Timedelta(days=1), count=sequential_returns + 2)
		returns = [get_rate_of_change(a.close, b.close) for a, b in zip(records, records[1:])]
		x[i] = returns[1:]
		y[i] = returns[0]