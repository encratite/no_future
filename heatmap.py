import warnings
from math import sqrt
from statistics import stdev, mean
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import quantile_transform

from common import read_ohlc_series, get_rate_of_change

VOLATILITY_DAYS: Final[int] = 20

def render_heatmap(symbol: str, start: pd.Timestamp, end: pd.Timestamp, x_axis: str, y_axis: str, quantiles: int) -> None:
	assert start < end
	assert quantiles >= 2
	series = read_ohlc_series(symbol)
	y_values: list[float] = []
	return1_values: list[float] = []
	return2_values: list[float] = []
	volume_values: list[float] = []
	open_interest_values: list[float] = []
	volatility_values: list[float] = []
	time = start
	while time < end:
		tomorrow = series.get(time + pd.Timedelta(days=1), right=True)
		records = series.get(time, count=VOLATILITY_DAYS + 1)
		today = records[0]
		yesterday = records[1]
		y = get_rate_of_change(tomorrow.close, today.close)
		returns = [get_rate_of_change(a.close, b.close) for a, b in zip(records[1:], records)]
		return1 = returns[0]
		return2 = returns[1]
		volume = get_rate_of_change(today.volume, yesterday.volume)
		open_interest = get_rate_of_change(today.open_interest, yesterday.open_interest)
		volatility = stdev(returns) * sqrt(VOLATILITY_DAYS)
		y_values.append(y)
		return1_values.append(return1)
		return2_values.append(return2)
		volume_values.append(volume)
		open_interest_values.append(open_interest)
		volatility_values.append(volatility)
		time += pd.Timedelta(days=1)
	return1_quantiles = get_quantile_transform(return1_values)
	return2_quantiles = get_quantile_transform(return2_values)
	volume_quantiles = get_quantile_transform(volume_values)
	open_interest_quantiles = get_quantile_transform(open_interest_values)
	volatility_quantiles = get_quantile_transform(volatility_values)
	values = {
		"return1": ("Previous day's returns", return1_quantiles),
		"return2": ("Day before yesterday's returns", return2_quantiles),
		"volume": ("Change in volume from yesterday", volume_quantiles),
		"interest": ("Change in open interest from yesterday", open_interest_quantiles),
		"volatility": (f"{VOLATILITY_DAYS}-Day volatility", volatility_quantiles)
	}
	x_axis_title, x_axis_values = values[x_axis]
	y_axis_title, y_axis_values = values[y_axis]
	mean_returns_matrix = np.zeros((quantiles, quantiles))
	annotations = np.empty((quantiles, quantiles), dtype=object)
	for i in range(quantiles):
		for j in range(quantiles):
			x_quantile_min, x_quantile_max = get_quantile_limits(i, quantiles)
			y_quantile_min, y_quantile_max = get_quantile_limits(j, quantiles)
			matching_y_values = []
			for k, y in enumerate(y_values):
				x_quantile = x_axis_values[k]
				y_quantile = y_axis_values[k]
				x_match = x_quantile_min < x_quantile < x_quantile_max
				y_match = y_quantile_min < y_quantile < y_quantile_max
				if x_match and y_match:
					matching_y_values.append(y)
			if len(matching_y_values) > 0:
				mean_returns = mean(matching_y_values)
			else:
				mean_returns = 0
			mean_returns_matrix[i, j] = mean_returns
			sample_count = len(matching_y_values)
			amount = float(sample_count) / len(y_values)
			if sample_count > 50:
				annotations[i, j] = f"{mean_returns:+.2%}\n({amount:.2%})"
			elif sample_count > 0:
				annotations[i, j] = f"{mean_returns:+.2%}\n{sample_count} samples"
			else:
				annotations[i, j] = f"No samples"
	plt.figure(figsize=(10, 10))
	tick_labels = [f"Quantile {i + 1}" for i in range(quantiles)]
	sns.heatmap(mean_returns_matrix, annot=annotations, fmt="", xticklabels=tick_labels, yticklabels=tick_labels)
	plt.title(f"Single Day Returns of {symbol} by Feature Quantiles\n(from {format_date(start)} to {format_date(end)})")
	plt.xlabel(x_axis_title)
	plt.ylabel(y_axis_title)
	plt.show()
	plt.close()

def format_date(time: pd.Timestamp) -> str:
	return time.strftime("%Y-%m-%d")

def get_quantile_limits(i: int, quantiles: int) -> tuple[float, float]:
	quantile_min = i / float(quantiles)
	quantile_max = (i + 1) / float(quantiles)
	return quantile_min, quantile_max

def get_quantile_transform(values: list[float]) -> npt.NDArray:
	warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing")
	array = np.array(values).reshape(-1, 1)
	output = quantile_transform(array)
	return output