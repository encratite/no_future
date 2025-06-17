from collections import defaultdict
from typing import cast
from functools import reduce

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from common import read_ohlc_series
from ohlc import OhlcRecord
from series import TimeSeries

def analyze_momentum(
	symbols: list[str],
	start: pd.Timestamp | None,
	end: pd.Timestamp | None
) -> None:
	symbol_series: dict[str, TimeSeries[OhlcRecord]] = {}
	for symbol in symbols:
		symbol_series[symbol] = read_ohlc_series(symbol)
	# analyze_momentum_horizon(1, 1, 100, 1, start, end, symbol_series)
	# analyze_momentum_horizon(5, 1, 200, 1, start, end, symbol_series)
	analyze_momentum_horizon(20, 1, 300, 1, start, end, symbol_series)
	# analyze_momentum_horizon(60, 1, 300, 1, start, end, symbol_series)

def analyze_momentum_horizon(
		forecast_horizon: int,
		momentum_start: int,
		momentum_end: int,
		momentum_step: int,
		start: pd.Timestamp | None,
		end: pd.Timestamp | None,
		symbol_series: dict[str, TimeSeries[OhlcRecord]]
) -> None:
	momentum_dfs = [get_momentum_horizon_data(
		symbol,
		forecast_horizon,
		momentum_start,
		momentum_end,
		momentum_step,
		start,
		end,
		series
	) for symbol, series in symbol_series.items()]
	id_var = "momentum"
	var_name = "Symbol"
	value_name = "correlation"
	merged_df = reduce(lambda left, right: pd.merge(left, right, on=id_var), momentum_dfs)
	symbols = list(symbol_series.keys())
	melted_df = merged_df.melt(
		id_vars=id_var,
		value_vars=symbols,
		var_name=var_name,
		value_name=value_name
	)
	plt.figure(figsize=(12, 8))
	sns.lineplot(melted_df, x=id_var, y=value_name, hue=var_name)
	plt.xlim(melted_df[id_var].min(), melted_df[id_var].max())
	plt.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
	plt.xlabel("Momentum (days)")
	plt.ylabel("Pearson's ρ")
	plt.title(f"Correlation between n-Day Momentum and {forecast_horizon}-Day Returns")
	plt.tight_layout()
	plt.show()
	plt.close()

def get_momentum_horizon_data(
		symbol: str,
		forecast_horizon: int,
		momentum_start: int,
		momentum_end: int,
		momentum_step: int,
		start: pd.Timestamp | None,
		end: pd.Timestamp | None,
		series: TimeSeries[OhlcRecord]
) -> pd.DataFrame:
	momentum_returns_dict: defaultdict[int, list[float]] = defaultdict(list)
	returns: list[float] = []
	records = series.values()
	i = momentum_end
	while i < len(records) - forecast_horizon:
		time = records[i].time
		if start is not None and time < start:
			i += 1
			continue
		if end is not None and time >= end:
			break
		today = records[i].close
		horizon = records[i + forecast_horizon].close
		horizon_returns = horizon / today - 1
		returns.append(horizon_returns)
		momentum = momentum_start
		while momentum <= momentum_end:
			momentum_close = records[i - momentum].close
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
		symbol: y_correlation
	})
	return df