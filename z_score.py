from multiprocessing import Pool
from statistics import mean, stdev
from time import perf_counter
from typing import Final

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backend_bases import MouseEvent

from common import (
	read_ohlc_series,
	get_rate_of_change,
	get_sharpe_ratio
)
from ohlc import OhlcRecord
from series import TimeSeries

UNADJUSTED_CLOSE_MINIMUM: Final[float] = 0.1
FREQUENCY_MINIMUM: Final[float] = 0.005

TIME: Final[str] = "time"
EQUITY: Final[str] = "equity"

def analyze_z_score_pattern(
	symbol1: str,
	symbol2: str,
	start: pd.Timestamp,
	end: pd.Timestamp,
	detailed: bool
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
	time_series, z_scores1, next_day_returns = output[0]
	_, z_scores2, _ = output[1]
	assert len(z_scores1) == len(next_day_returns)
	assert len(z_scores1) == len(z_scores2), f"{len(z_scores1)} != {len(z_scores2)}"
	regular_boundaries = [
		(None, -0.75),
		(-0.75, 0.0),
		(0.0, 0.75),
		(0.75, None),
	]
	detailed_boundaries = [
		(None, -1.25),
		(-1.25, -0.75),
		(-0.75, 0.0),
		(0.0, 0.75),
		(0.75, 1.25),
		(1.25, None)
	]
	boundaries = detailed_boundaries if detailed else regular_boundaries
	n = len(boundaries)
	returns_matrix: list[list[list[tuple[pd.Timestamp, float]]]] = [[[] for _ in range(n)] for _ in range(n)]
	for time, z_score1, z_score2, returns in zip(time_series, z_scores1, z_scores2, next_day_returns):
		cell1 = get_cell(z_score1, boundaries)
		cell2 = get_cell(z_score2, boundaries)
		returns_matrix[cell1][cell2].append((time, returns))
	sharpe_ratio_matrix = np.zeros((n, n))
	annotations = np.empty((n, n), dtype=object)
	for i in range(n):
		for j in range(n):
			time_returns = returns_matrix[i][j]
			returns = [x[1] for x in time_returns]
			frequency = len(returns) / len(next_day_returns)
			if len(returns) >= 2 and frequency > FREQUENCY_MINIMUM:
				sharpe_ratio = get_sharpe_ratio(returns)
				annotation = f"{sharpe_ratio:.2f}\n({frequency:.2%})"
			else:
				sharpe_ratio = 0
				annotation = "-"
			sharpe_ratio_matrix[i, j] = sharpe_ratio
			annotations[i, j] = annotation
	duration = perf_counter() - perf_start
	print(f"Calculated Z-score matrix in {duration:.2f} s")
	fig, ax = plt.subplots(figsize=(12, 8))

	def onclick(event: MouseEvent) -> None:
		if event.inaxes == ax:
			x, y = int(event.xdata), int(event.ydata)
			data = returns_matrix[y][x]
			show_equity_curve(data, x, y)

	fig.canvas.mpl_connect("button_press_event", onclick) # type: ignore
	tick_labels = []
	for lower_boundary, upper_boundary in boundaries:
		if lower_boundary is not None and upper_boundary is not None:
			label = f"{lower_boundary} to {upper_boundary}"
		elif lower_boundary is None:
			label = f"< {upper_boundary}"
		else:
			label = f"> {lower_boundary}"
		tick_labels.append(label)
	ax = sns.heatmap(sharpe_ratio_matrix, ax=ax, annot=annotations, fmt="", xticklabels=tick_labels, yticklabels=tick_labels)
	cbar = ax.collections[0].colorbar
	formatter = ticker.FuncFormatter(lambda x, _: f"{x:.2f}")
	cbar.ax.yaxis.set_major_formatter(formatter)
	plt.title(f"{symbol1} vs. {symbol2}")
	plt.xlabel(symbol2)
	plt.ylabel(symbol1)
	plt.show()
	plt.close()

def show_equity_curve(time_returns: list[tuple[pd.Timestamp, float]], x: int, y: int) -> None:
	time_series = [x[0] for x in time_returns]
	cash = 1
	equity_curve = []
	for _time, returns in time_returns:
		cash *= 1 + returns
		equity_curve.append(cash)
	df = pd.DataFrame({
		TIME: time_series,
		EQUITY: equity_curve
	})
	fig, ax = plt.subplots(figsize=(12, 8))
	sns.lineplot(df, ax=ax, x=TIME, y=EQUITY, label="Capital")
	ax.legend().set_visible(False)
	ax.set_xlim(df[TIME].min(), df[TIME].max())
	ax.set_xlabel("Date")
	ax.set_ylabel("Capital")
	ax.set_title(f"Equity Curve of Z-Score Cell ({x + 1}, {y + 1})")
	plt.tight_layout()
	fig.show()

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
) -> tuple[list[pd.Timestamp], list[float], list[float]]:
	records = series.values()
	returns = [get_rate_of_change(a.unadjusted_close, b.unadjusted_close) for a, b in zip(records[1:], records)]
	z_scores = []
	next_day_returns = []
	past_returns_mean: float | None = None
	past_returns_sigma: float | None = None
	last_update: pd.Timestamp | None = None
	time_series = []
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
		time_series.append(record.time)
		z_scores.append(z_score)
		next_day_returns.append(tomorrow)
	return time_series, z_scores, next_day_returns

def get_cell(z_score: float, boundaries: list[tuple[float | None, float | None]]) -> int:
	for i, boundary_tuple in enumerate(boundaries):
		lower_boundary, upper_boundary = boundary_tuple
		if (lower_boundary is None or z_score >= lower_boundary) and (upper_boundary is None or z_score < upper_boundary):
			return i
	raise Exception(f"Unknown to determine cell for z-score {z_score}")