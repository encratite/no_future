from collections import defaultdict
import os
from typing import Any, cast

import pandas as pd

from common import read_ohlc_series
from constant import OHLC_PRECISION
from configuration import Configuration
from globex import GlobexCode
from ohlc import OhlcRecord
from contracts import get_barchart_symbol

def generate_intraday_contract(symbol: str) -> None:
	series = read_ohlc_series(symbol)
	barchart_symbol = get_barchart_symbol(symbol)
	input_path = os.path.join(Configuration.BARCHART_DIRECTORY, f"{barchart_symbol}.H1.csv")
	df: pd.DataFrame = pd.read_csv(input_path, parse_dates=["time"])
	intraday_records: defaultdict[pd.Timestamp, defaultdict[GlobexCode, list[OhlcRecord]]] = defaultdict(lambda: defaultdict(list))
	for row in df.itertuples():
		row = cast(Any, row)
		record = OhlcRecord(row)
		intraday_records[record.time.normalize()][record.globex_code].append(record)
	df_dict: defaultdict[str, list[Any]] = defaultdict(list)
	for record in series.values():
		offset = record.close - record.unadjusted_close
		matching_day_records = intraday_records[record.time][record.globex_code]
		for intraday_record in matching_day_records:
			df_dict["time"].append(intraday_record.time.date())
			df_dict["symbol"].append(repr(intraday_record.globex_code))
			df_dict["open"].append(round(intraday_record.open + offset, OHLC_PRECISION))
			df_dict["high"].append(round(intraday_record.high + offset, OHLC_PRECISION))
			df_dict["low"].append(round(intraday_record.low + offset, OHLC_PRECISION))
			df_dict["close"].append(round(intraday_record.close + offset, OHLC_PRECISION))
			df_dict["unadjusted_close"].append(intraday_record.close)
			df_dict["volume"].append(intraday_record.volume)
			df_dict["open_interest"].append(intraday_record.open_interest)
	df = pd.DataFrame(df_dict)
	output_path = os.path.join(Configuration.FEATHER_INTRADAY_DIRECTORY, f"{symbol}.feather")
	df.to_feather(output_path)