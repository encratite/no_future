from statistics import stdev
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import read_ohlc_series, get_log_returns

TIME: Final[str] = "time"
VOLATILITY: Final[str] = "volatility"

def render_volatility_chart(symbol: str, window_size: int, start: pd.Timestamp, end: pd.Timestamp) -> None:
	assert window_size >= 2
	series = read_ohlc_series(symbol)
	records = series.values()
	time_series = []
	volatility_values = []
	for i, record in enumerate(records):
		if record.time < start:
			continue
		elif record.time > end:
			break
		offset = i + 1
		window = records[offset - window_size - 1: offset]
		closes = [x.close for x in window]
		returns = [get_log_returns(a, b) for a, b in zip(closes[1:], closes)]
		assert len(returns) == window_size
		volatility = stdev(returns)
		time_series.append(record.time)
		volatility_values.append(volatility)
	df = pd.DataFrame({
		TIME: time_series,
		VOLATILITY: volatility_values
	})
	plt.figure(figsize=(12, 8))
	sns.lineplot(x=df[TIME], y=df[VOLATILITY], label=symbol)
	plt.xlim(df[TIME].min(), df[TIME].max())
	plt.title(f"Volatility of {symbol} ({window_size} Days)")
	plt.xlabel("Time")
	plt.ylabel("Volatility")
	plt.legend()
	plt.tight_layout()
	plt.show()
	plt.close()