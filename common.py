import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import log
from typing import Any, Callable, Iterable, TypeVar

import matplotlib.dates as mdates
import numpy as np
from colorama import Fore, Style
from tabulate import tabulate
from tqdm import tqdm

from configuration import Configuration
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

def print_table(table: list[list[Any]], headers: bool = True, always_right: bool = False):
	numeric_columns = len(table[0]) - 1
	if always_right:
		column_alignment = len(table[0]) * ("right",)
	else:
		column_alignment = ("left",) + numeric_columns * ("right",)
	headers = "firstrow" if headers else []
	table_string = tabulate(table, headers=headers, tablefmt="simple_outline", disable_numparse=True, colalign=column_alignment)
	print(table_string)
	print("")

def read_ohlc_series(symbol: str) -> TimeSeries[OhlcRecord]:
	if "." not in symbol:
		file_name = f"{symbol}.F1"
	else:
		file_name = symbol
	path = os.path.join(Configuration.FEATHER_DIRECTORY, f"{file_name}.feather")
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