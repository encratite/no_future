from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import read_ohlc_series

ID_VAR: Final[str] = "time"
VAR_NAME: Final[str] = "value"
VALUE_NAME: Final[str] = "value_name"

def render_comparison_chart(
	symbols: list[str],
	start: pd.Timestamp,
	end: pd.Timestamp
) -> None:
	series = [read_series(symbol, start, end) for symbol in symbols]
	time_series = series[0][1]
	closes = [normalize_closes(series_closes) for series_closes, _ in series]
	min_length = len(time_series)
	for series_closes in closes:
		min_length = min(len(series_closes), min_length)
	time_series = time_series[:min_length]
	closes = [series_closes[:min_length] for series_closes in closes]
	df_dict: dict[str, list[pd.Timestamp | float]] = {
		ID_VAR: time_series
	}
	for i, symbol in enumerate(symbols):
		series_closes = closes[i]
		df_dict[symbol] = series_closes
	df = pd.DataFrame(df_dict)
	df_melted = df.melt(
		id_vars=ID_VAR,
		value_vars=symbols,
		var_name=VAR_NAME,
		value_name=VALUE_NAME
	)
	_fig, ax = plt.subplots(figsize=(12, 8))
	sns.lineplot(df_melted, ax=ax, x=ID_VAR, y=VALUE_NAME, hue=VAR_NAME)
	x_min = df_melted[ID_VAR].min()
	x_max = df_melted[ID_VAR].max()
	plt.xlim(x_min, x_max)
	plt.xlabel("Day")
	plt.ylabel("Relative price")
	plt.title(", ".join(symbols))
	ax.legend().set_title(None) # type: ignore
	plt.tight_layout()
	plt.show()
	plt.close()

def read_series(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[list[float], list[pd.Timestamp]]:
	series = read_ohlc_series(symbol)
	records = [x for x in series.values() if start <= x.time < end]
	time_series = [x.time for x in records]
	closes = [x.close for x in records]
	return closes, time_series

def normalize_closes(closes: list[float]) -> list[float]:
	return [x / closes[0] for x in closes]