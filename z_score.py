from statistics import mean, stdev
from multiprocessing import Pool
from time import perf_counter
from typing import Final

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from common import read_ohlc_series, get_log_returns
from ohlc import OhlcRecord
from series import TimeSeries

UNADJUSTED_CLOSE_MINIMUM: Final[float] = 0.1
# FREQUENCY_MINIMUM: Final[float] = 0.005
FREQUENCY_MINIMUM: Final[float] = 0

def analyze_z_score_pattern(
	symbol1: str,
	symbol2: str,
	start: pd.Timestamp,
	end: pd.Timestamp
) -> None:
	perf_start = perf_counter()
	series1 = read_ohlc_series(symbol1)
	series2 = read_ohlc_series(symbol2)
	time_series1 = get_filtered_time_series(series1, start, end)
	time_series2 = get_filtered_time_series(series2, start, end)
	shared_time_series = time_series1 & time_series2
	arguments = [
		(series1, shared_time_series),
		(series2, shared_time_series)
	]
	with Pool(2) as pool:
		output = list(pool.starmap(get_momentum2_z_scores, arguments))
	z_scores1, next_day_returns = output[0]
	z_scores2, _ = output[1]
	assert len(z_scores1) == len(next_day_returns)
	assert len(z_scores1) == len(z_scores2), f"{len(z_scores1)} != {len(z_scores2)}"
	boundaries = [
		(None, -1.25),
		(-1.25, -0.75),
		(-0.75, 0.0),
		(0.0, 0.75),
		(0.75, 1.25),
		(1.25, None)
	]
	n = len(boundaries)
	matrix = [[[] for _ in range(n)] for _ in range(n)]
	for z_score1, z_score2, returns in zip(z_scores1, z_scores2, next_day_returns):
		cell1 = get_cell(z_score1, boundaries)
		cell2 = get_cell(z_score2, boundaries)
		matrix[cell1][cell2].append(returns)
	mean_returns_matrix = np.zeros((n, n))
	annotations = np.empty((n, n), dtype=object)
	for i in range(n):
		for j in range(n):
			returns = matrix[i][j]
			frequency = len(returns) / len(next_day_returns)
			if len(returns) > 0 and frequency > FREQUENCY_MINIMUM:
				mean_returns = mean(returns)
				annotation = f"{mean_returns:.2%}\n({frequency:.2%})"
			else:
				mean_returns = 0
				annotation = "-"
			mean_returns_matrix[i, j] = mean_returns
			annotations[i, j] = annotation
	duration = perf_counter() - perf_start
	print(f"Calculated Z-score matrix in {duration:.2f} s")
	plt.figure(figsize=(12, 8))
	tick_labels = []
	for lower_boundary, upper_boundary in boundaries:
		if lower_boundary is not None and upper_boundary is not None:
			label = f"{lower_boundary} to {upper_boundary}"
		elif lower_boundary is None:
			label = f"< {upper_boundary}"
		else:
			label = f"> {lower_boundary}"
		tick_labels.append(label)
	ax = sns.heatmap(mean_returns_matrix, annot=annotations, fmt="", xticklabels=tick_labels, yticklabels=tick_labels)
	cbar = ax.collections[0].colorbar
	formatter = ticker.FuncFormatter(lambda x, _: f"{x * 100:.2f}%")
	cbar.ax.yaxis.set_major_formatter(formatter)
	plt.title(f"{symbol1} vs. {symbol2}")
	plt.xlabel(symbol2)
	plt.ylabel(symbol1)
	plt.show()
	plt.close()

def get_filtered_time_series(
	series: TimeSeries[OhlcRecord],
	start: pd.Timestamp,
	end: pd.Timestamp
) -> set[pd.Timestamp]:
	time_series = [x.time for x in series.values() if x.time >= start and x.time < end and (x.globex_code.root == "J6" or x.unadjusted_close > UNADJUSTED_CLOSE_MINIMUM)]
	return set(time_series)

def get_momentum2_z_scores(
	series: TimeSeries[OhlcRecord],
	shared_time_series: set[pd.Timestamp]
) -> tuple[list[float], list[float]]:
	records = series.values()
	returns = [get_log_returns(a.unadjusted_close, b.unadjusted_close) for a, b in zip(records[1:], records)]
	z_scores = []
	next_day_returns = []
	past_returns_mean: float | None = None
	past_returns_sigma: float | None = None
	last_update: pd.Timestamp | None = None
	for i, record in enumerate(records[1:-1]):
		if record.time not in shared_time_series:
			continue
		today = returns[i]
		tomorrow = returns[i + 1]
		if last_update is None or record.time.month != last_update.month:
			past_returns = returns[:i]
			past_returns_mean = mean(past_returns)
			past_returns_sigma = stdev(past_returns)
		z_score = (today - past_returns_mean) / past_returns_sigma
		z_scores.append(z_score)
		next_day_returns.append(tomorrow)
	return z_scores, next_day_returns

def get_cell(z_score: float, boundaries: list[tuple[float | None, float | None]]) -> int:
	for i, boundary_tuple in enumerate(boundaries):
		lower_boundary, upper_boundary = boundary_tuple
		if (lower_boundary is None or z_score >= lower_boundary) and (upper_boundary is None or z_score < upper_boundary):
			return i
	raise Exception(f"Unknown to determine cell for z-score {z_score}")