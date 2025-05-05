from statistics import mean, stdev

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from common import read_ohlc_series, get_log_returns
from ohlc import OhlcRecord
from series import TimeSeries

def analyze_z_score_pattern(
	symbol1: str,
	symbol2: str,
	start: pd.Timestamp,
	end: pd.Timestamp,
	anchor: pd.Timestamp | None
) -> None:
	series1 = read_ohlc_series(symbol1)
	series2 = read_ohlc_series(symbol2)
	time_series1 = set(series1)
	time_series2 = set(series2)
	shared_time_series = time_series1 & time_series2
	z_scores1, next_day_returns = get_momentum2_z_scores(series1, start, end, anchor, shared_time_series)
	z_scores2, _ = get_momentum2_z_scores(series2, start, end, anchor, shared_time_series)
	assert len(z_scores1) == len(next_day_returns)
	assert len(z_scores1) == len(z_scores2)
	boundaries = [
		(None, -1.5),
		(-1.5, -1.0),
		(-1.0, 0.0),
		(0.0, 1.00),
		(1.0, 1.5),
		(1.5, None)
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
			if len(returns) > 0:
				mean_returns = mean(returns)
			else:
				mean_returns = 0
			mean_returns_matrix[i, j] = mean_returns
			if len(returns) > 0:
				annotation = f"{mean_returns:.2%}\n({len(returns) / len(next_day_returns):.2%})"
			else:
				annotation = "-"
			annotations[i, j] = annotation
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

def get_momentum2_z_scores(
	series: TimeSeries[OhlcRecord],
	start: pd.Timestamp,
	end: pd.Timestamp,
	anchor: pd.Timestamp | None,
	shared_time_series: set[pd.Timestamp]
) -> tuple[list[float], list[float]]:
	records = series.values()
	records = [x for x in records if (anchor is None or x.time >= anchor) and x.time < end]
	returns = [get_log_returns(a.close, b.close) for a, b in zip(records[1:], records)]
	z_scores = []
	next_day_returns = []
	for i, record in enumerate(records[1:]):
		if record.time < start or record.time not in shared_time_series:
			continue
		if record.time >= end:
			break
		today = returns[i]
		past_returns = returns[:i]
		z_score = (today - mean(past_returns)) / stdev(past_returns)
		z_scores.append(z_score)
		next_day_returns.append(today)
	return z_scores, next_day_returns

def get_cell(z_score: float, boundaries: list[tuple[float | None, float | None]]) -> int:
	for i, boundary_tuple in enumerate(boundaries):
		lower_boundary, upper_boundary = boundary_tuple
		if (lower_boundary is None or z_score >= lower_boundary) and (upper_boundary is None or z_score < upper_boundary):
			return i
	raise Exception(f"Unknown to determine cell for z-score {z_score}")