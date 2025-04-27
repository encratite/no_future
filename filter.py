import pandas as pd

from globex import GlobexCode
from ohlc import OhlcRecord

class ContractFilter:
	# Only set if the exchange symbol is different from the Barchart symbol
	exchange_symbol: str | None
	# Copy the contract to this symbol (convenient for micro contracts)
	copy: str | None
	# Limits the number to Fn records, such that 1 <= n <= f_records_limit
	f_records_limit: int | None
	# Enables the generation of records for the nearest contract that is at least one year behind the front contract
	enable_fy_records: bool
	# If set, any Globex code prior to this one will be excluded from continuous contract generation
	_legacy_cutoff: GlobexCode | None
	# If set, the include/exclude filters aren't applied until this Globex code has been reached
	_first_filter_contract: GlobexCode | None
	# If set, the filters aren't applied anymore once this Globex code has been reached
	_last_filter_contract: GlobexCode | None
	# If set, then only these months will be included
	_include_months: list[str] | None
	# If set, then all of these months will be excluded
	_exclude_months: list[str] | None
	# If set, acts as an additional cutoff filter (to avoid corrupted entries)
	_cutoff_date: pd.Timestamp | None

	def __init__(
		self,
		exchange_symbol: str | None = None,
		copy: str | None = None,
		f_records_limit: int | None = None,
		enable_fy_records: bool = True,
		legacy_cutoff: str | None = None,
		first_filter_contract: str | None = None,
		last_filter_contract: str | None = None,
		include_months: list[str] | None = None,
		exclude_months: list[str] | None = None,
		cutoff_date: str | None = None
	) -> None:
		self.exchange_symbol = exchange_symbol
		self.copy = copy
		self.f_records_limit = f_records_limit
		self.enable_fy_records = enable_fy_records
		self._legacy_cutoff = GlobexCode(legacy_cutoff) if legacy_cutoff is not None else None
		self._first_filter_contract = GlobexCode(first_filter_contract) if first_filter_contract is not None else None
		self._last_filter_contract = GlobexCode(last_filter_contract) if last_filter_contract is not None else None
		self._include_months = include_months
		self._exclude_months = exclude_months
		self._cutoff_date = pd.Timestamp(cutoff_date)
		if include_months is not None and exclude_months is not None:
			raise Exception("Invalid filter configuration")

	def include_record(self, record: OhlcRecord) -> bool:
		if self._cutoff_date is not None and record.time < self._cutoff_date:
			return False
		globex_code = record.globex_code
		if self._legacy_cutoff is not None and globex_code < self._legacy_cutoff:
			return False
		if self._first_filter_contract is not None and self._last_filter_contract is not None:
			if not (self._first_filter_contract <= globex_code < self._last_filter_contract):
				return True
		elif self._first_filter_contract is not None and globex_code < self._first_filter_contract:
			return True
		elif self._last_filter_contract is not None and globex_code >= self._last_filter_contract:
			return True
		if self._include_months is not None:
			return globex_code.month in self._include_months
		elif self._exclude_months is not None:
			return globex_code.month not in self._exclude_months
		else:
			return True