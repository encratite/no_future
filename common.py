import os
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Iterator

from colorama import Fore, Style
from tabulate import tabulate

from configuration import Configuration
from ohlc import OhlcRecord
from series import TimeSeries

def execute_thread_pool(fn: Any, *iterable: Iterable) -> Iterator[Any]:
	if sysconfig.get_config_vars()["Py_GIL_DISABLED"] == 1:
		thread_count = os.cpu_count()
	else:
		thread_count = 1
	with ThreadPoolExecutor(max_workers=thread_count) as executor:
		return executor.map(fn, *iterable)

def get_performance_string(performance: float) -> str:
	return format_percentage(performance - 1)

def format_percentage(percentage: float) -> str:
	output = f"{percentage:+.2%}"
	if percentage > 0:
		output = f"{Fore.GREEN}{output}{Style.RESET_ALL}"
	elif percentage < 0:
		output = f"{Fore.RED}{output}{Style.RESET_ALL}"
	return output

def print_table(table: list[list[Any]]):
	numeric_columns = len(table[0]) - 1
	column_alignment = ("left",) + numeric_columns * ("right",)
	table_string = tabulate(table, headers="firstrow", tablefmt="simple_outline", disable_numparse=True, colalign=column_alignment)
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