import warnings
from statistics import mean

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import seaborn as sns
from sklearn.preprocessing import quantile_transform

from constant import DAYS_PER_YEAR
from ohlc import OhlcRecord
from series import TimeSeries
from .data import HeatmapData
from .t_test import perform_t_test

def render_quantile_data(
	symbol: str,
	start: pd.Timestamp,
	end: pd.Timestamp,
	x_axis: str,
	y_axis: str,
	quantiles: int,
	statistics_only: bool,
	data: HeatmapData,
	series: TimeSeries[OhlcRecord]
) -> None:
	values = data.get_values()
	x_axis_title, x_axis_values = values[x_axis]
	y_axis_title, y_axis_values = values[y_axis]
	perform_t_test(x_axis, y_axis, x_axis_values, y_axis_values, data.y_values, series, start, end)
	if statistics_only:
		return
	mean_returns_matrix = np.zeros((quantiles, quantiles))
	annotations = np.empty((quantiles, quantiles), dtype=object)
	for i in range(quantiles):
		for j in range(quantiles):
			x_quantile_min, x_quantile_max = get_quantile_limits(i, quantiles)
			y_quantile_min, y_quantile_max = get_quantile_limits(j, quantiles)
			matching_y_values = []
			for k, y in enumerate(data.y_values):
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
			y_values_array = np.array(matching_y_values)
			skew = scipy.stats.skew(y_values_array)
			kurtosis = scipy.stats.kurtosis(y_values_array)
			mean_returns_matrix[i, j] = mean_returns
			trades = len(matching_y_values)
			years_traded = (end - start).days / DAYS_PER_YEAR
			trades_per_year = trades / years_traded
			if trades > 0:
				annotations[i, j] = f"Mean: {mean_returns:+.2%}\nSkew: {skew:.2f}\nKurtosis: {kurtosis:.2f}\nTrades: {trades_per_year:.1f}"
			else:
				annotations[i, j] = f"No samples"
	plt.figure(figsize=(12, 8))
	tick_labels = [f"Quantile {i + 1}" for i in range(quantiles)]
	ax = sns.heatmap(mean_returns_matrix, annot=annotations, fmt="", xticklabels=tick_labels, yticklabels=tick_labels)
	cbar = ax.collections[0].colorbar
	formatter = ticker.FuncFormatter(lambda x, _: f"{x * 100:.2f}%")
	cbar.ax.yaxis.set_major_formatter(formatter)
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