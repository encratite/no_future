import glob
import os
import re
from collections import defaultdict
from copy import copy
import time
from typing import Any, cast

import pandas as pd

from common import execute_pool
from configuration import Configuration
from contracts import get_contract_filter
from filter import ContractFilter
from globex import GlobexCode
from ohlc import OhlcRecord

def generate_contract(symbol: str) -> None:
	input_path = os.path.join(Configuration.BARCHART_DIRECTORY, f"{symbol}.D1.csv")
	df: pd.DataFrame = pd.read_csv(input_path, parse_dates=["time"])
	contracts_per_day: dict[pd.Timestamp, list[OhlcRecord]] = defaultdict(list)
	contract_ranges: dict[GlobexCode, tuple[pd.Timestamp, pd.Timestamp]] = {}
	contract_filter = get_contract_filter(symbol)
	for row in df.itertuples():
		row = cast(Any, row)
		record = OhlcRecord(row)
		if record.time < Configuration.CUTOFF_DATE:
			# The record is too old, skip it
			continue
		if contract_filter is not None and not contract_filter.include_record(record):
			# The contract filter caused the record to be excluded
			continue
		contracts_per_day[row.time].append(record)
		if row.symbol in contract_ranges:
			first, last = contract_ranges[record.globex_code]
			contract_ranges[record.globex_code] = (min(row.time, first), max(row.time, last))
		else:
			contract_ranges[record.globex_code] = (row.time, row.time)
	# Maps continuous contract keys to tuples of OHLC records and close offsets
	# For example, keys 1, 2, 3 respectively represent the F1, F2, F3 continuous contracts
	record_offsets: dict[str, list[tuple[OhlcRecord, float]]] = defaultdict(list)
	# Skip old records
	time_records = [(time, records) for time, records in contracts_per_day.items()]
	# Technically redundant, but make sure the records are sorted by time either way
	time_records = sorted(time_records, key=get_time_key)
	calculate_rollover_offsets(time_records, contract_ranges, record_offsets, contract_filter)
	if contract_filter is not None and contract_filter.exchange_symbol is not None:
		exchange_symbol = contract_filter.exchange_symbol
	else:
		exchange_symbol = symbol
	write_contract_files(exchange_symbol, record_offsets)
	if contract_filter.copy is not None:
		write_contract_files(contract_filter.copy, record_offsets)

def get_time_key(time_records: tuple[pd.Timestamp, list[OhlcRecord]]) -> pd.Timestamp:
	time, _records = time_records
	return time

def calculate_rollover_offsets(
	time_records: list[tuple[pd.Timestamp, list[OhlcRecord]]],
	contract_ranges: dict[GlobexCode, tuple[pd.Timestamp, pd.Timestamp]],
	record_offsets: dict[str, list[tuple[OhlcRecord, float]]],
	contract_filter: ContractFilter | None
):
	_, first_records = next(iter(time_records))
	# Select first Globex code by open interest
	first_record = get_record_by_open_interest(first_records)
	current_globex_code = first_record.globex_code
	f1_key = "F1"
	# Calculate rollovers and keep track of the offsets
	for time, records in time_records:
		current_record = next((x for x in records if x.globex_code == current_globex_code), None)
		if current_record is None:
			raise Exception(f"Unable to find a record for current contract {current_globex_code} at {time.date()}")
		_, last_contract_day = contract_ranges[current_globex_code]
		filtered_records = [x for x in records if is_rollover_target(x, time, last_contract_day, current_record, current_globex_code)]
		offset = 0
		if len(filtered_records) > 0:
			new_record = get_record_by_open_interest(filtered_records)
			if new_record.globex_code > current_globex_code:
				# Roll over into new contract
				offset = new_record.close - current_record.close
				current_globex_code = new_record.globex_code
				current_record = new_record
		record_offsets[f1_key].append((current_record, offset))
		f_records = generate_f_records(current_globex_code, records, record_offsets, contract_filter)
		generate_fy_records(current_globex_code, f_records, record_offsets, contract_filter)

def generate_f_records(
	current_globex_code: GlobexCode,
	records: list[OhlcRecord],
	record_offsets: dict[str, list[tuple[OhlcRecord, float]]],
	contract_filter: ContractFilter | None
) -> list[OhlcRecord]:
	# Create records for contracts F2, F3, etc.
	f_records = [x for x in records if x.globex_code > current_globex_code]
	f_records = sorted(f_records, key=lambda x: x.globex_code)
	f_number = 2
	for record in f_records:
		if contract_filter is not None and contract_filter.f_records_limit is not None:
			if f_number > cast(int, contract_filter.f_records_limit):
				break
		f_key = f"F{f_number}"
		previous_records = record_offsets[f_key]
		offset = 0
		if len(previous_records) > 0:
			last_record, _last_offset = previous_records[-1]
			if record.globex_code > last_record.globex_code:
				offset = record.close - last_record.close
			elif record.globex_code < last_record.globex_code:
				# Awkward case, an unexpectedly low Globex code that wasn't previously available
				# Ignore it
				continue
		record_offsets[f_key].append((record, offset))
		f_number += 1
		if f_number > Configuration.CONTINUOUS_CONTRACT_F_LIMIT:
			break
	return f_records

def generate_fy_records(
	current_globex_code: GlobexCode,
	f_records: list[OhlcRecord],
	record_offsets: dict[str, list[tuple[OhlcRecord, float]]],
	contract_filter: ContractFilter | None
) -> None:
	if contract_filter is not None and not contract_filter.enable_fy_records:
		return
	# Create records for FY contract (i.e. the contract one year ahead of F1)
	fy_globex_code = copy(current_globex_code)
	fy_globex_code.add_year()
	fy_records = [x for x in f_records if x.globex_code >= fy_globex_code]
	fy_key = "FY"
	if len(fy_records) > 0:
		# Not strictly correct since it might select a record that is even further out than a year
		record = fy_records[0]
		previous_records = record_offsets[fy_key]
		if len(previous_records) > 0:
			last_record, _last_offset = previous_records[-1]
			if record.globex_code > last_record.globex_code:
				offset = record.close - last_record.close
				record_offsets[fy_key].append((record, offset))
			elif record.globex_code == last_record.globex_code:
				record_offsets[fy_key].append((record, 0))
		else:
			record_offsets[fy_key].append((record, 0))

def write_contract_files(symbol: str, record_offsets: dict[str, list[tuple[OhlcRecord, float]]]) -> None:
	for f_key, records in record_offsets.items():
		# Calculate global offset from differences between contracts, in reverse
		global_offset = 0
		for i in reversed(range(len(records))):
			record, offset = records[i]
			records[i] = record, global_offset
			global_offset += offset
		# Generate DataFrame using the Panama canal method
		df_dict = defaultdict(list)
		# This is extremely questionable and previously caused horrible bugs
		# Who the hell would use floats to represent monetary values anyway?
		precision = 6
		for record, offset in records:
			df_dict["time"].append(record.time.date())
			df_dict["symbol"].append(repr(record.globex_code))
			df_dict["open"].append(round(record.open + offset, precision))
			df_dict["high"].append(round(record.high + offset, precision))
			df_dict["low"].append(round(record.low + offset, precision))
			df_dict["close"].append(round(record.close + offset, precision))
			df_dict["unadjusted_close"].append(record.close)
			df_dict["volume"].append(record.volume)
			df_dict["open_interest"].append(record.open_interest)
		df = pd.DataFrame(df_dict)
		output_path = os.path.join(Configuration.FEATHER_DIRECTORY, f"{symbol}.{f_key}.feather")
		df.to_feather(output_path)

def is_rollover_target(
	record: OhlcRecord,
	time: pd.Timestamp,
	last_contract_day: pd.Timestamp,
	current_record: OhlcRecord,
	current_globex_code: GlobexCode
) -> bool:
	last_day = time == last_contract_day
	enough_open_interest = record.open_interest > current_record.open_interest > 0
	return record.globex_code > current_globex_code and (last_day or enough_open_interest)

def get_record_by_open_interest(records: list[OhlcRecord]) -> OhlcRecord:
	return max(records, key=lambda x: x.open_interest)

def generate_all_contracts():
	glob_pattern = os.path.join(Configuration.BARCHART_DIRECTORY, "*.D1.csv")
	paths = glob.glob(glob_pattern)
	basename_pattern = re.compile("^[A-Z0-9]+")
	symbols = []
	for path in paths:
		basename = os.path.basename(path)
		match = basename_pattern.match(basename)
		if match is None:
			continue
		symbol = match[0]
		symbols.append(symbol)
	execute_pool(generate_contract, symbols)