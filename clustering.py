from collections import defaultdict
from itertools import permutations
from math import prod
from multiprocessing import Pool, cpu_count
from statistics import mean
from time import perf_counter
from typing import Final, TypeAlias

import numpy as np
import pandas as pd
from colorama import Fore, Style
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from common import (
	get_rate_of_change,
	get_sharpe_ratio,
	print_table,
	format_percentage
)
from configuration import Configuration
from constant import DAYS_PER_YEAR
from manager import AssetManager
from strategy import Strategy

FeatureCache: TypeAlias = dict[str, tuple[list[list[float]], list[float]]]

ENABLE_MULTIPROCESSING: Final[bool] = True
ENABLE_DELTA_ANALYSIS: Final[bool] = False
ENABLE_SHARPE_RATIO: Final[bool] = False
SAMPLES_MINIMUM: Final[float] = 0.03
SHARPE_RATIOS_MINIMUM_COUNT: Final[int] = 3
ENABLE_WEIGHTS: Final[bool] = True
FEATURE_WEIGHTS: Final[list[float]] = [
	# 2.0, 1.0, 1.0, 1.0
	# 1.0, 0.0, 0.0, 0.0
	2.0, 0, 0, 1.0
]

class ClusterResults:
	mean_annual_return: float
	sharpe_ratio: float | None
	cluster_size: float

	def __init__(
		self,
		mean_annual_return: float,
		sharpe_ratio: float | None,
		cluster_size: float
	) -> None:
		self.mean_annual_return = mean_annual_return
		self.sharpe_ratio = sharpe_ratio
		self.cluster_size = cluster_size

class SymbolResults:
	symbol1: str
	symbol2: str
	clusters: list[ClusterResults]

	def __init__(
		self,
		symbol1: str,
		symbol2: str,
		clusters: list[ClusterResults]
	):
		self.symbol1 = symbol1
		self.symbol2 = symbol2
		self.clusters = clusters

def analyze_clusters(symbols: list[str], clusters: int, start: pd.Timestamp, end: pd.Timestamp) -> None:
	assert clusters >= 2
	assert start < end
	perf_start = perf_counter()
	if ENABLE_MULTIPROCESSING:
		processes = cpu_count()
		with Pool(processes) as pool:
			arguments = [(symbols, clusters, start, end, x, processes) for x in range(processes)]
			output = pool.starmap(execute_analysis, arguments)
		all_results: list[SymbolResults] = []
		for results in output:
			all_results += results
	else:
		all_results = execute_analysis(symbols, clusters, start, end, 0, 1)
	all_results = sorted(all_results, key=lambda x: f"{x.symbol1} {x.symbol2}")
	headers = ["Symbol 1", "Symbol 2"]
	headers += [f"MR {x + 1}" for x in range(clusters)]
	if ENABLE_SHARPE_RATIO:
		headers += [f"SR {x + 1}" for x in range(clusters)]
	if ENABLE_DELTA_ANALYSIS:
		headers.append("Delta")
	headers += [f"Size {x + 1}" for x in range(clusters)]
	table = [headers]
	for results in all_results:
		row = [
			results.symbol1,
			results.symbol2
		]
		for cluster in results.clusters:
			mean_returns_string = format_percentage(cluster.mean_annual_return)
			row.append(mean_returns_string)
		if ENABLE_SHARPE_RATIO:
			for cluster in results.clusters:
				sharpe_ratio_string = get_sharpe_ratio_string(cluster.sharpe_ratio)
				row.append(sharpe_ratio_string)
		delta: float | None = None
		sharpe_ratios = [x.sharpe_ratio for x in results.clusters if x.sharpe_ratio is not None] # type: ignore
		if len(sharpe_ratios) >= SHARPE_RATIOS_MINIMUM_COUNT:
			minimum_sharpe_ratio = min(sharpe_ratios)
			maximum_sharpe_ratio = max(sharpe_ratios)
			delta = abs(maximum_sharpe_ratio - minimum_sharpe_ratio)
		delta_string = get_sharpe_ratio_string(delta)
		if ENABLE_DELTA_ANALYSIS:
			row.append(delta_string)
		for cluster in results.clusters:
			row.append(f"{cluster.cluster_size:.1%}")
		table.append(row)
	print_table(table)
	if ENABLE_DELTA_ANALYSIS:
		print_sharpe_ratio_deltas(all_results)
	duration = perf_counter() - perf_start
	print(f"Finished analysis in {duration:.2f} s")

def print_sharpe_ratio_deltas(all_results: list[SymbolResults]) -> None:
	deltas: defaultdict[str, list[float]] = defaultdict(list)
	for results in all_results:
		sharpe_ratios = [x.sharpe_ratio for x in results.clusters if x.sharpe_ratio is not None]
		if len(sharpe_ratios) < SHARPE_RATIOS_MINIMUM_COUNT:
			continue
		minimum_sharpe_ratio = min(sharpe_ratios)
		maximum_sharpe_ratio = max(sharpe_ratios)
		delta = abs(maximum_sharpe_ratio - minimum_sharpe_ratio)
		deltas[results.symbol1].append(delta)
	headers = ["Symbol", "Sharpe Ratio Delta"]
	table = [headers]
	mean_deltas = []
	for symbol, symbol_deltas in deltas.items():
		mean_delta = mean(symbol_deltas)
		row = [
			symbol,
			get_sharpe_ratio_string(mean_delta)
		]
		table.append(row)
		mean_deltas.append(mean_delta)
	total_mean_delta = mean(mean_deltas)
	row = [
		"Mean",
		get_sharpe_ratio_string(total_mean_delta)
	]
	table.append(row)
	print_table(table)

def get_sharpe_ratio_string(sharpe_ratio: float | None) -> str:
	if sharpe_ratio is not None:
		sharpe_ratio_string = f"{sharpe_ratio:.2f}"
		if sharpe_ratio > 1.0:
			sharpe_ratio_string = f"{Fore.GREEN}{sharpe_ratio_string}{Style.RESET_ALL}"
		if sharpe_ratio < -1.0:
			sharpe_ratio_string = f"{Fore.CYAN}{sharpe_ratio_string}{Style.RESET_ALL}"
		elif sharpe_ratio < 0.0:
			sharpe_ratio_string = f"{Fore.RED}{sharpe_ratio_string}{Style.RESET_ALL}"
	else:
		sharpe_ratio_string = "-"
	return sharpe_ratio_string

def execute_analysis(
	symbols: list[str],
	clusters: int,
	start: pd.Timestamp,
	end: pd.Timestamp,
	process_id: int,
	processes: int
) -> list[SymbolResults]:
	asset_manager = AssetManager(symbols)
	symbol_permutations = permutations(symbols, 2)
	output = []
	feature_cache: FeatureCache = {}
	for i, permutation in enumerate(symbol_permutations):
		if i % processes == process_id:
			symbol1 = permutation[0]
			symbol2 = permutation[1]
			results = analyze_pair(symbol1, symbol2, clusters, start, end, feature_cache, asset_manager)
			results.process_id = process_id
			output.append(results)
	return output

def analyze_pair(
	symbol1: str,
	symbol2: str,
	clusters: int,
	start: pd.Timestamp,
	end: pd.Timestamp,
	feature_cache: FeatureCache,
	asset_manager: AssetManager
) -> SymbolResults:
	features1, labels = get_features(symbol1, start, end, feature_cache, asset_manager)
	features2, _ = get_features(symbol2, start, end, feature_cache, asset_manager)
	features = features1 + features2
	x = np.array(features).T
	scaler = StandardScaler()
	x = scaler.fit_transform(x)
	k_means = KMeans(n_clusters=clusters, random_state=Configuration.SEED)
	if ENABLE_WEIGHTS:
		feature_weights = np.array(FEATURE_WEIGHTS)
		x = x * np.concatenate((feature_weights, feature_weights))
	k_means.fit(x)
	cluster_returns: defaultdict[int, list[float]] = defaultdict(list)
	for i, label in enumerate(k_means.labels_):
		returns = labels[i]
		cluster_returns[label].append(returns)
	all_cluster_results: list[ClusterResults] = []
	years = (end - start) / pd.Timedelta(days=DAYS_PER_YEAR)
	for i in range(clusters):
		mean_annual_return = 0.0
		sharpe_ratio = None
		cluster_size = 0.0
		if i in cluster_returns:
			returns = cluster_returns[i]
			total_return = prod(1 + x for x in returns) - 1
			mean_annual_return = total_return / years
			cluster_size = len(returns) / len(labels)
			if len(returns) / len(labels) >= SAMPLES_MINIMUM:
				sharpe_ratio = get_sharpe_ratio(returns)
		cluster_results = ClusterResults(mean_annual_return, sharpe_ratio, cluster_size)
		all_cluster_results.append(cluster_results)
	all_cluster_results = sorted(all_cluster_results, key=lambda cluster: cluster.mean_annual_return, reverse=True)
	results = SymbolResults(symbol1, symbol2, all_cluster_results)
	return results

def get_features(
	symbol: str,
	start: pd.Timestamp,
	end: pd.Timestamp,
	feature_cache: FeatureCache,
	asset_manager: AssetManager
) -> tuple[list[list[float]], list[float]]:
	if symbol in feature_cache:
		return feature_cache[symbol]
	reference = asset_manager.get_series("ES")
	series = asset_manager.get_series(symbol)
	momentum2_values = []
	momentum3_values = []
	momentum5_values = []
	momentum10_values = []
	labels = []
	for time in reference:
		if time < start:
			continue
		if time >= end:
			break
		tomorrow = series.get(time + pd.Timedelta(days=1), right=True)
		records = series.get(time, count=10)
		closes = [x.close for x in records]
		all_closes = closes + [tomorrow.close]
		if records[0].time != time or any(x <= 0 for x in all_closes) or Strategy.is_banned_symbol(symbol, time):
			momentum2_values.append(0)
			momentum3_values.append(0)
			momentum5_values.append(0)
			momentum10_values.append(0)
			labels.append(0)
			continue

		def get_momentum(days: int) -> float:
			return get_rate_of_change(closes[0], closes[days - 1])

		momentum2 = get_momentum(2)
		momentum3 = get_momentum(3)
		momentum5 = get_momentum(5)
		momentum10 = get_momentum(10)
		returns = get_rate_of_change(tomorrow.close, closes[0])
		momentum2_values.append(momentum2)
		momentum3_values.append(momentum3)
		momentum5_values.append(momentum5)
		momentum10_values.append(momentum10)
		labels.append(returns)
	features = [
		momentum2_values,
		momentum3_values,
		momentum5_values,
		momentum10_values,
	]
	feature_cache[symbol] = (features, labels)
	return features, labels