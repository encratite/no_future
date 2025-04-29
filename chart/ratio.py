from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import read_ohlc_series

ID_VAR: Final[str] = "time"
VAR_NAME: Final[str] = "value"
VALUE_NAME: Final[str] = "value_name"
BASE_VAR: Final[str] = "Base"
RATIO_VAR: Final[str] = "Ratio"

def render_ratio_chart(
	base_symbol: str,
	dividend_symbol: str,
	divisor_symbol: str,
	start: pd.Timestamp,
	end: pd.Timestamp
) -> None:
	base_closes, time_series = read_series(base_symbol, start, end)
	normalized_base = [x / base_closes[0] for x in base_closes]
	dividend_closes, _ = read_series(dividend_symbol, start, end)
	divisor_closes, _ = read_series(divisor_symbol, start, end)
	ratios = [a / b for a, b in zip(dividend_closes, divisor_closes)]
	ratios = [x / ratios[0] for x in ratios]
	min_length = min(len(time_series), len(ratios), len(normalized_base))
	df = pd.DataFrame({
		ID_VAR: time_series[:min_length],
		BASE_VAR: normalized_base[:min_length],
		RATIO_VAR: ratios[:min_length]
	})
	df_melted = df.melt(
		id_vars=ID_VAR,
		value_vars=[BASE_VAR, RATIO_VAR],
		var_name=VAR_NAME,
		value_name=VALUE_NAME
	)
	_fig, ax = plt.subplots(figsize=(12, 8))
	sns.lineplot(df_melted, ax=ax, x=ID_VAR, y=VALUE_NAME, hue=VAR_NAME)
	x_min = df_melted[ID_VAR].min()
	x_max = df_melted[ID_VAR].max()
	plt.xlim(x_min, x_max)
	plt.xlabel("Day")
	plt.ylabel("Value")
	plt.title(f"{base_symbol} vs. {dividend_symbol} / {divisor_symbol}")
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