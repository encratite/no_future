import calendar
from collections import defaultdict
from enum import Enum
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import read_ohlc_series
from strategy import Strategy

ID_VAR: Final[str] = "index"
VAR_NAME: Final[str] = "value"
VALUE_NAME: Final[str] = "value_name"

class SeasonalityChartMode(Enum):
	DAY_OF_WEEK: Final[int] = 0
	MONTH: Final[int] = 1
	QUARTER: Final[int] = 2

def render_seasonality_chart(symbol: str, mode: SeasonalityChartMode, start: pd.Timestamp, end: pd.Timestamp) -> None:
	assert start < end
	df_melted = get_seasonality_chart_data(symbol, mode, start, end)
	_fig, ax = plt.subplots(figsize=(12, 8))
	sns.lineplot(df_melted, ax=ax, x=ID_VAR, y=VALUE_NAME, hue=VAR_NAME)
	x_min = df_melted[ID_VAR].min()
	x_max = df_melted[ID_VAR].max()
	plt.xlim(x_min, x_max)
	plt.xlabel("Day")
	plt.ylabel("Price")
	plt.title(f"Cumulative Seasonality of {symbol}")
	ax.legend().set_title(None) # type: ignore
	plt.tight_layout()
	plt.show()
	plt.close()

def get_seasonality_chart_data(symbol: str, mode: SeasonalityChartMode, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
	series = read_ohlc_series(symbol)
	records = series.values()
	records = [x for x in records if start <= x.time < end and not Strategy.is_banned_symbol(symbol, x.time)]
	time_returns: list[tuple[pd.Timestamp, float]] = [(a.time, b.close / a.close - 1) for a, b in zip(records, records[1:])]
	returns_dict: defaultdict[str, list[float]] = defaultdict(list)
	for time, returns in time_returns:
		match mode:
			case SeasonalityChartMode.DAY_OF_WEEK:
				key = calendar.day_name[time.day_of_week]
			case SeasonalityChartMode.MONTH:
				key = calendar.month_name[time.month]
			case SeasonalityChartMode.QUARTER:
				key = f"Q{time.month // 3 + 1}"
			case _:
				raise unknown_mode()
		returns_dict[key].append(returns)
	minimum_length: int | None = None
	for returns in returns_dict.values():
		length = len(returns)
		if minimum_length is None:
			minimum_length = length
		else:
			minimum_length = min(minimum_length, length)
	for key in returns_dict:
		returns = returns_dict[key]
		returns_dict[key] = returns[:minimum_length]
	prices_dict: defaultdict[str, list[float]] = defaultdict(list)
	for key, returns in returns_dict.items():
		prices_dict[key] = get_prices_from_returns(returns)
	for i in range(minimum_length + 1):
		prices_dict[ID_VAR].append(i + 1)
	df = pd.DataFrame(prices_dict)
	match mode:
		case SeasonalityChartMode.DAY_OF_WEEK:
			value_vars = list(calendar.day_name)[:5]
		case SeasonalityChartMode.MONTH:
			value_vars = list(calendar.month_name)[1:]
		case SeasonalityChartMode.QUARTER:
			value_vars = [f"Q{x + 1}" for x in range(4)]
		case _:
			raise unknown_mode()
	df_melted = df.melt(
		id_vars=ID_VAR,
		value_vars=value_vars,
		var_name=VAR_NAME,
		value_name=VALUE_NAME
	)
	return df_melted

def unknown_mode() -> Exception:
	return Exception("Unknown seasonality chart mode")

def get_prices_from_returns(seasonal_returns: list[float]) -> list[float]:
	synthetic_price = 10.0
	prices = [synthetic_price]
	for returns in seasonal_returns:
		synthetic_price *= 1 + returns
		prices.append(synthetic_price)
	return prices