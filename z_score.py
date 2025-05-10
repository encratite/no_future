from math import sqrt
from multiprocessing import Pool
from statistics import mean, stdev
from time import perf_counter
from typing import Final, Callable
from types import CodeType

from colorama import Fore, Style
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backend_bases import MouseEvent
from scipy.stats import pearsonr

from common import (
	read_ohlc_series,
	get_rate_of_change,
	get_mean_annual_return,
	print_table
)
from ohlc import OhlcRecord
from series import TimeSeries

UNADJUSTED_CLOSE_MINIMUM: Final[float] = 0.1
FREQUENCY_MINIMUM: Final[float] = 0.005
ANALYSIS_WINDOW_SIZE: Final[int] = 20
EXTREME_STATS_RATIO: Final[float] = 0.2
MINIMUM_SAMPLES: Final[int] = 10

TIME: Final[str] = "time"
EQUITY: Final[str] = "equity"

class AnalysisStats:
	correlation: float
	volatility1: float
	volatility2: float

	def __init__(
		self,
		correlation: float,
		volatility1: float,
		volatility2: float
	) -> None:
		self.correlation = correlation
		self.volatility1 = volatility1
		self.volatility2 = volatility2

def analyze_z_score_pattern(
	symbol1: str,
	symbol2: str,
	start: pd.Timestamp,
	end: pd.Timestamp,
	detailed: bool,
	delay: bool,
	boundary: float | None,
	minimum: float | None,
	maximum: float | None
) -> None:
	perf_start = perf_counter()
	series1 = read_ohlc_series(symbol1)
	series2 = read_ohlc_series(symbol2)
	time_series1 = get_filtered_time_series(series1, start, end)
	time_series2 = get_filtered_time_series(series2, start, end)
	shared_time_series = time_series1 & time_series2
	arguments = [
		(series1, None, shared_time_series, delay),
		(series2, None, shared_time_series, delay),
		(series1, series2, shared_time_series, delay)
	]
	with Pool(len(arguments)) as pool:
		output = list(pool.starmap(pool_proxy, arguments))
	time_series, z_scores1, next_day_returns = output[0]
	_, z_scores2, _ = output[1]
	stats: dict[pd.Timestamp, AnalysisStats] = output[2]
	if delay:
		time_series = time_series[1:]
		z_scores1 = z_scores1[1:]
		next_day_returns = next_day_returns[1:]
		z_scores2 = z_scores2[:-1]
	assert len(z_scores1) == len(next_day_returns)
	assert len(z_scores1) == len(z_scores2), f"{len(z_scores1)} != {len(z_scores2)}"
	if boundary is None:
		boundary = 0.4
	regular_boundaries = [
		(None, -boundary),
		(-boundary, boundary),
		(boundary, None),
	]
	detailed_boundaries = [
		(None, -boundary),
		(-boundary, 0.0),
		(0.0, boundary),
		(boundary, None),
	]
	boundaries = detailed_boundaries if detailed else regular_boundaries
	n = len(boundaries)
	returns_matrix: list[list[list[tuple[pd.Timestamp, float]]]] = [[[] for _ in range(n)] for _ in range(n)]
	for time, z_score1, z_score2, returns in zip(time_series, z_scores1, z_scores2, next_day_returns):
		cell1 = get_cell(z_score1, boundaries)
		cell2 = get_cell(z_score2, boundaries)
		if minimum is not None and (abs(z_score1) < minimum or abs(z_score2) < minimum):
			continue
		if maximum is not None and (abs(z_score1) > maximum or abs(z_score2) > maximum):
			continue
		returns_matrix[cell1][cell2].append((time, returns))
	annual_returns_matrix = np.zeros((n, n))
	annotations = np.empty((n, n), dtype=object)
	for i in range(n):
		for j in range(n):
			time_returns = returns_matrix[i][j]
			returns = [x[1] for x in time_returns]
			mean_annual_return = 0
			annotation = "-"
			if len(next_day_returns) > 0:
				frequency = len(returns) / len(next_day_returns)
				if len(returns) >= 2 and frequency > FREQUENCY_MINIMUM:
					mean_annual_return, risk_adjusted_return = get_return_metrics(returns, start, end)
					annotation = f"{mean_annual_return:.2%} MAR\n{risk_adjusted_return:.2f} RAR\n({frequency:.2%})"
			annual_returns_matrix[i, j] = mean_annual_return
			annotations[i, j] = annotation
	duration = perf_counter() - perf_start
	mean_annual_return = get_mean_annual_return(next_day_returns, start, end)
	risk_adjusted_return = mean(next_day_returns) / stdev(next_day_returns)
	print(f"Buy and hold MAR: {mean_annual_return:.2%}")
	print(f"Buy and hold RAR: {risk_adjusted_return:.2f}")
	print(f"Calculated Z-score matrix in {duration:.2f} s")
	fig, ax = plt.subplots(figsize=(12, 8))
	active_cell: tuple[int, int] | None = None

	def onclick(event: MouseEvent) -> None:
		if event.inaxes == ax:
			x, y = int(event.xdata), int(event.ydata)
			nonlocal active_cell
			active_cell = x, y
			data = returns_matrix[y][x]
			show_equity_curve(symbol1, symbol2, start, end, data, x, y, stats)

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
	ax = sns.heatmap(annual_returns_matrix, ax=ax, annot=annotations, fmt="", xticklabels=tick_labels, yticklabels=tick_labels)
	cbar = ax.collections[0].colorbar
	formatter = ticker.FuncFormatter(lambda x, _: f"{x:.2f}")
	cbar.ax.yaxis.set_major_formatter(formatter)
	info = []
	if minimum is not None:
		info.append(f"minimum {minimum}")
	if maximum is not None:
		info.append(f"maximum {maximum}")
	info_string = ""
	if len(info) > 0:
		info_string = f" ({", ".join(info)})"
	plt.title(f"{symbol1} vs. {symbol2} Momentum Z-Scores{info_string}")
	plt.xlabel(symbol2)
	plt.ylabel(symbol1)
	plt.show(block=False)
	while True:
		filter_expression = input()
		if filter_expression == "":
			break
		try:
			filter_code = compile(filter_expression, "<string>", "eval")
			execute_filter(0, 0, 0, filter_code)
		except Exception as error:
			print(error)
			continue
		x, y = active_cell
		active_data = returns_matrix[y][x]
		show_equity_curve(symbol1, symbol2, start, end, active_data, x, y, stats, filter_code)
	plt.close()

def execute_filter(
	correlation: float,
	volatility1: float,
	volatility2: float,
	filter_code: CodeType
) -> bool:
	parameters = {
		"correlation": correlation,
		"volatility1": volatility1,
		"volatility2": volatility2
	}
	output = eval(filter_code, {}, parameters)
	if not isinstance(output, bool) and not isinstance(output, np.bool):
		raise Exception("Return type of filter must be bool")
	return output

def show_equity_curve(
	symbol1: str,
	symbol2: str,
	start: pd.Timestamp,
	end: pd.Timestamp,
	time_returns: list[tuple[pd.Timestamp, float]],
	x: int,
	y: int,
	stats: dict[pd.Timestamp, AnalysisStats],
	filter_code: CodeType | None = None
) -> None:
	if filter_code is not None:
		filtered_timestamps: set[pd.Timestamp] = set()
		filtered_stats: dict[pd.Timestamp, AnalysisStats] = {}
		for time, analysis_stats in stats.items():
			accepted = execute_filter(
				analysis_stats.correlation,
				analysis_stats.volatility1,
				analysis_stats.volatility2,
				filter_code
			)
			if accepted:
				filtered_timestamps.add(time)
				filtered_stats[time] = analysis_stats
		removed_ratio = 1 - len(filtered_stats) / len(stats)
		print(f"Filter removed {removed_ratio:.2%} of samples")
		time_returns = [(time, returns) for time, returns in time_returns if time in filtered_timestamps]
		stats = filtered_stats
		if len(time_returns) < MINIMUM_SAMPLES:
			print(f"Not enough samples left ({len(time_returns)})")
			return
	analyze_cell_returns(symbol1, symbol2, start, end, time_returns, stats)
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

def analyze_cell_returns(
	symbol1: str,
	symbol2: str,
	start: pd.Timestamp,
	end: pd.Timestamp,
	time_returns: list[tuple[pd.Timestamp, float]],
	stats: dict[pd.Timestamp, AnalysisStats]
) -> None:
	positive_stats_returns = []
	negative_stats_returns = []
	for time, returns in time_returns:
		analysis_stats = stats[time]
		target = positive_stats_returns if returns > 0 else negative_stats_returns
		target.append((analysis_stats, returns))
	positive_stats, very_positive_stats = convert_stats(positive_stats_returns, True)
	negative_stats, very_negative_stats = convert_stats(negative_stats_returns)
	table = [
		["Statistic", "Profits", "Losses", f"Best {EXTREME_STATS_RATIO:.0%} of Profits", f"Worst {EXTREME_STATS_RATIO:.0%} of Losses"],
		[f"Number of samples"],
		[f"Number of samples (%)"],
		[f"Correlation of {symbol1} and {symbol2}"],
		[f"Volatility of {symbol1}"],
		[f"Volatility of {symbol2}"],
		[f"Correlation of {symbol1} and {symbol2} (Z-score)"],
		[f"Volatility of {symbol1} (Z-score)"],
		[f"Volatility of {symbol2} (Z-score)"]
	]
	groups = [
		positive_stats,
		negative_stats,
		very_positive_stats,
		very_negative_stats
	]
	stats_values = list(stats.values())
	for group in groups:
		correlation_mean, correlation_z_score = get_mean_and_z_score(lambda x: x.correlation, group, stats_values)
		volatility1_mean, volatility1_z_score = get_mean_and_z_score(lambda x: x.volatility1, group, stats_values)
		volatility2_mean, volatility2_z_score = get_mean_and_z_score(lambda x: x.volatility2, group, stats_values)
		cells = [
			f"{len(group)}",
			f"{len(group) / len(time_returns):.2%}",
			format_value(correlation_mean),
			format_value(volatility1_mean),
			format_value(volatility2_mean),
			format_value(correlation_z_score),
			format_value(volatility1_z_score),
			format_value(volatility2_z_score),
		]
		for i, cell in enumerate(cells):
			table[i + 1].append(cell)
	print_table(table, newline=False)
	returns = [r for t, r in time_returns]
	mean_annual_return, risk_adjusted_return = get_return_metrics(returns, start, end)
	print(f"{mean_annual_return:.2%} MAR, {risk_adjusted_return:.2f} RAR\n")

def get_return_metrics(
	returns: list[float],
	start: pd.Timestamp,
	end: pd.Timestamp,
) -> tuple[float, float]:
	mean_annual_return = get_mean_annual_return(returns, start, end)
	risk_adjusted_return = mean(returns) / stdev(returns)
	return mean_annual_return, risk_adjusted_return

def format_value(value: float) -> str:
	if value > 0:
		return f"{value:.3f}"
	else:
		return f"{Fore.RED}{value:.3f}{Style.RESET_ALL}"

def get_mean_and_z_score(select: Callable[[AnalysisStats], float], group: list[AnalysisStats], stats: list[AnalysisStats]) -> tuple[float, float]:
	group_values = [select(x) for x in group]
	all_values = [select(x) for x in stats]
	group_mean = mean(group_values)
	all_mean = mean(all_values)
	all_sigma = stdev(all_values)
	z_score = (group_mean - all_mean) / all_sigma
	return group_mean, z_score

def convert_stats(stats_returns: list[tuple[AnalysisStats, float]], reverse: bool = False) -> tuple[list[AnalysisStats], list[AnalysisStats]]:
	group_stats_returns = sorted(stats_returns, key=lambda x: x[1], reverse=reverse)
	group = [x[0] for x in group_stats_returns]
	extreme_group_samples = round(EXTREME_STATS_RATIO * len(group_stats_returns))
	extreme_group = [x[0] for x in group_stats_returns[:extreme_group_samples]]
	return group, extreme_group

def get_filtered_time_series(
	series: TimeSeries[OhlcRecord],
	start: pd.Timestamp,
	end: pd.Timestamp
) -> set[pd.Timestamp]:
	time_series = [x.time for x in series.values() if x.time >= start and x.time < end and (x.globex_code.root in ["J6", "M6"] or x.unadjusted_close > UNADJUSTED_CLOSE_MINIMUM)]
	return set(time_series)

def pool_proxy(
	series1: TimeSeries[OhlcRecord],
	series2: TimeSeries[OhlcRecord] | None,
	shared_time_series: set[pd.Timestamp],
	delay: bool
) -> tuple[list[pd.Timestamp], list[float], list[float]] | dict[pd.Timestamp, AnalysisStats]:
	if series2 is None:
		return get_momentum2_z_scores(series1, shared_time_series)
	else:
		return get_correlation(series1, series2, shared_time_series, delay)

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

def get_correlation(
	series1: TimeSeries[OhlcRecord],
	series2: TimeSeries[OhlcRecord] | None,
	shared_time_series: set[pd.Timestamp],
	delay: bool
) -> dict[pd.Timestamp, AnalysisStats]:
	output: dict[pd.Timestamp, AnalysisStats] = {}
	for time in shared_time_series:
		def get_returns(series: TimeSeries[OhlcRecord], simulate_delay: bool) -> list[float]:
			adjusted_time = time
			if simulate_delay:
				adjusted_time = time - pd.Timedelta(days=1)
			records = series.get(adjusted_time, count=ANALYSIS_WINDOW_SIZE)
			returns = [get_rate_of_change(a.unadjusted_close, b.unadjusted_close) for a, b in zip(records[1:], records)]
			return returns
		returns1 = get_returns(series1, False)
		returns2 = get_returns(series2, delay)
		correlation = pearsonr(returns1, returns2).statistic
		volatility_factor = sqrt(len(returns1))
		volatility1 = volatility_factor * stdev(returns1)
		volatility2 = volatility_factor * stdev(returns2)
		stats = AnalysisStats(correlation, volatility1, volatility2)
		output[time] = stats
	return output