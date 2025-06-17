from typing import Final

import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

from common import read_ohlc_series, get_rate_of_change

TIME: Final[str] = "time"
COEFFICIENT: Final[str] = "coefficient"

def render_correlation_chart(
		symbol1: str,
		symbol2: str,
		sliding_window: int,
		start: pd.Timestamp,
		end: pd.Timestamp
) -> None:
	def get_daily_returns(symbol: str) -> dict[pd.Timestamp, float]:
		series = read_ohlc_series(symbol)
		records = [x for x in series.values() if start <= x.time < end]
		output = {}
		for a, b in zip(records[1:], records):
			returns = get_rate_of_change(a.close, b.close)
			output[a.time] = returns
		return output
	returns_dict1 = get_daily_returns(symbol1)
	returns_dict2 = get_daily_returns(symbol2)
	times = []
	returns1 = []
	returns2 = []
	for time, returns in returns_dict1.items():
		if time in returns_dict2:
			times.append(time)
			returns1.append(returns)
			returns2.append(returns_dict2[time])
	assert(0 < sliding_window < len(returns1))
	coefficients = []
	for i in range(sliding_window, len(returns1)):
		range1 = returns1[i - sliding_window:i]
		range2 = returns2[i - sliding_window:i]
		statistic = pearsonr(range1, range2).statistic
		coefficients.append(statistic)
	df = pd.DataFrame({
		TIME: times[sliding_window:],
		COEFFICIENT: coefficients
	})
	plt.figure(figsize=(12, 8))
	sns.lineplot(data=df, x=TIME, y=COEFFICIENT)
	plt.xlim(df[TIME].min(), df[TIME].max())
	plt.xlabel("Time")
	plt.ylabel("Pearson's ρ")
	plt.title(f"Sliding window PCC for {symbol1} and {symbol2} ({sliding_window} days)")
	plt.tight_layout()
	plt.show()
	plt.close()