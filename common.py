import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import log, sqrt, prod
from statistics import mean, stdev
from typing import Any, Callable, Iterable, TypeVar

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from colorama import Fore, Style
from tabulate import tabulate
from tqdm import tqdm

from asset import Asset
from configuration import Configuration
from constant import DAYS_PER_YEAR, TRADING_DAYS_PER_YEAR
from ohlc import OhlcRecord
from series import TimeSeries

A = TypeVar("A")
B = TypeVar("B")

def execute_pool(fn: Callable[[A], B], iterables: Iterable[A]) -> list[B]:
	with ProcessPoolExecutor() as executor:
		futures = [executor.submit(fn, x) for x in iterables]
		output = list(tqdm(as_completed(futures), total=len(futures), colour="green"))
	return output

def get_performance_string(performance: float) -> str:
	return format_percentage(performance - 1)

def format_percentage(percentage: float) -> str:
	output = f"{percentage:.2%}"
	if percentage > 0:
		output = f"{Fore.GREEN}{output}{Style.RESET_ALL}"
	elif percentage < 0:
		output = f"{Fore.RED}{output}{Style.RESET_ALL}"
	return output

def format_money(amount: float, console: bool = True) -> str:
	output = f"${amount:,.2f}"
	if amount < 0:
		if console:
			output = f"{Fore.RED}(${- amount:,.2f}){Style.RESET_ALL}"
		else:
			output = f"(${- amount:,.2f})"
	return output

def format_ratio(ratio: float | None) -> str:
	if ratio is None:
		return "-"
	ratio_string = f"{ratio:.2f}"
	if ratio >= 1:
		return f"{Fore.GREEN}{ratio_string}{Style.RESET_ALL}"
	elif ratio >= 0:
		return ratio_string
	else:
		return f"{Fore.RED}{ratio_string}{Style.RESET_ALL}"

def print_table(table: list[list[Any]], headers: bool = True, always_right: bool = False, newline: bool = True):
	numeric_columns = len(table[0]) - 1
	if always_right:
		column_alignment = len(table[0]) * ("right",)
	else:
		column_alignment = ("left",) + numeric_columns * ("right",)
	headers = "firstrow" if headers else []
	table_string = tabulate(table, headers=headers, tablefmt="simple_outline", disable_numparse=True, colalign=column_alignment)
	print(table_string)
	if newline:
		print("")

def read_ohlc_series(symbol: str, intraday: bool = False) -> TimeSeries[OhlcRecord]:
	if "." not in symbol and not intraday:
		file_name = f"{symbol}.F1"
	else:
		file_name = symbol
	if intraday:
		directory = Configuration.FEATHER_INTRADAY_DIRECTORY
	else:
		directory = Configuration.FEATHER_DIRECTORY
	path = os.path.join(directory, f"{file_name}.feather")
	ohlc_series = TimeSeries.read_ohlc_feather(path)
	return ohlc_series

def get_rate_of_change(new_value: float, old_value: float) -> float:
	return new_value / old_value - 1

def get_log_returns(new_value: float, old_value: float) -> float:
	value = new_value / old_value
	if value <= 0:
		value = 0.01
	return log(value)

def format_coord(x: float, y: float, ax: Any, format_string: Callable[[float], str] | None = None) -> str:
	date = mdates.num2date(x).strftime("%Y-%m-%d")
	nearest_points = []
	lines = ax.get_lines()
	for line in lines:
		xdata = line.get_xdata()
		ydata = line.get_ydata()
		if len(xdata) > 0:
			idx = np.abs(xdata - x).argmin()
			nearest_y = ydata[idx]
			label = line.get_label()
			if label.startswith("_"):
				continue
			if format_string is not None:
				y_string = format_string(nearest_y)
			else:
				y_string = f"{nearest_y:.2f}"
			nearest_points.append(f"{label}: {y_string}")
	if len(nearest_points) > 0:
		return f"[{date}] " + ", ".join(nearest_points)
	else:
		if format_string is not None:
			y_string = format_string(y)
		else:
			y_string = f"{y:.2f}"
		return f"[{date}] {y_string}"

def try_parse_int(int_string: str) -> int | None:
	try:
		return int(int_string)
	except ValueError:
		return None

def get_volatility(records: list[OhlcRecord]) -> float:
	records = sorted(records, key=lambda x: x.time)
	closes = [x.close for x in records]
	returns = [get_log_returns(a, b) for a, b in zip(closes[1:], closes)]
	volatility = stdev(returns)
	return volatility

def get_sharpe_ratio(returns: list[float]) -> float:
	mean_daily_returns = mean(returns)
	daily_standard_deviation = stdev(returns)
	mean_annual_returns = TRADING_DAYS_PER_YEAR * mean_daily_returns
	standard_deviation_factor = sqrt(TRADING_DAYS_PER_YEAR)
	standard_deviation = standard_deviation_factor * daily_standard_deviation
	sharpe_ratio = mean_annual_returns / standard_deviation
	return sharpe_ratio

def get_mean_annual_return(returns: list[float], start: pd.Timestamp, end: pd.Timestamp) -> float:
	years = (end - start) / pd.Timedelta(days=DAYS_PER_YEAR)
	total_return = prod(1 + x for x in returns) - 1
	mean_annual_return = total_return / years
	return mean_annual_return

def get_round_trip_cost_ratio(records: list[OhlcRecord], asset: Asset, additional_spread: int = 0) -> float:
	spread_ticks = asset.tick_size + additional_spread
	last_close = records[-1].close
	round_trip_cost = 2 * (asset.broker_fee + asset.exchange_fee + spread_ticks * asset.tick_value)
	nominal_value = last_close / asset.tick_size * asset.tick_value
	round_trip_cost_ratio = round_trip_cost / nominal_value
	return round_trip_cost_ratio