import os
import platform
import re
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Iterator
from tabulate import tabulate

from colorama import Fore, Style

def translate_path(path: str) -> str:
	if platform.system() == "Windows":
		return path
	elif platform.system() == "Linux":
		# Translate a Windows path to a WSL path
		pattern = re.compile(r"^(?P<drive>[A-Z]):\\(?P<path>.+)")
		match = pattern.match(path)
		if match is None:
			raise Exception(f"Invalid Windows path: {path}")
		drive = match.group("drive").lower()
		windows_path = match.group("path").replace("\\", "/")
		output = f"/mnt/{drive}/{windows_path}"
		return output
	else:
		raise Exception("Unknown operating system")

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