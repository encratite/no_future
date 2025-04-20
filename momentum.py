import os
from collections import defaultdict
from typing import cast

from scipy.stats import spearmanr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from common import read_ohlc_series
from ohlc import OhlcRecord
from series import TimeSeries

def analyze_momentum(symbol: str) -> None:
	records = read_ohlc_series(symbol)
	analyze_momentum_horizon(1, 1, 50, 1, symbol, records)
	analyze_momentum_horizon(5, 5, 200, 5, symbol, records)
	analyze_momentum_horizon(20, 20, 300, 10, symbol, records)

def analyze_momentum_horizon(
		forecast_horizon: int,
		momentum_start: int,
		momentum_end: int,
		momentum_step: int,
		symbol: str,
		records: TimeSeries[OhlcRecord]
) -> None:
	momentum_returns_dict: defaultdict[int, list[float]] = defaultdict(list)
	returns: list[float] = []
	closes = [cast(OhlcRecord, x).close for x in records.values()]
	i = momentum_end
	while i < len(closes) - forecast_horizon:
		today = closes[i]
		horizon = closes[i + forecast_horizon]
		horizon_returns = horizon / today - 1
		returns.append(horizon_returns)
		momentum = momentum_start
		while momentum <= momentum_end:
			momentum_close = closes[i - momentum]
			momentum_returns = today / momentum_close - 1
			momentum_returns_dict[momentum].append(momentum_returns)
			momentum += momentum_step
		i += 1
	x_momentum = list(momentum_returns_dict.keys())
	y_correlation: list[float] = []
	for momentum, momentum_returns in momentum_returns_dict.items():
		correlation = spearmanr(momentum_returns, returns).statistic # type: ignore
		y_correlation.append(correlation)
	df = pd.DataFrame({
		"momentum": x_momentum,
		"correlation": y_correlation
	})
	plt.figure(figsize=(12, 8))
	sns.lineplot(df, x="momentum", y="correlation")
	plt.xlim(df["momentum"].min(), df["momentum"].max())
	plt.ylim(min(df["correlation"].min() - 0.005, 0), max(df["correlation"].max() + 0.005, 0))
	plt.xlabel("Momentum (days)")
	plt.ylabel("Pearson's ρ")
	plt.title(f"Correlation between n-Day Momentum and {forecast_horizon}-Day Returns in {symbol}")
	plt.tight_layout()
	plt.show()
	plt.close()