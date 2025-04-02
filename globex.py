from __future__ import annotations

import re
from functools import total_ordering
from typing import Final

@total_ordering
class GlobexCode:
	GLOBEX_CODE_PATTERN: Final[re.Pattern] = re.compile("^([A-Z0-9]{2,})([FGHJKMNQUVXZ])([0-9]{2})$")

	symbol: str | None
	root: str
	month: str
	year: int

	def __init__(
			self,
			symbol: str,
			root: str | None = None,
			month: str | None = None,
			year: int | None = None
	) -> None:
		if root is not None and month is not None and year is not None:
			self.symbol = symbol
			self.root = root
			self.month = month
			self.year = year
		else:
			match = self.GLOBEX_CODE_PATTERN.match(symbol)
			if match is None:
				raise Exception("Invalid Globex code")
			self.symbol = symbol
			self.root = match[1]
			self.month = match[2]
			year = int(match[3])
			if year < 70:
				year += 1900
			else:
				year += 2000
			self.year = year

	@staticmethod
	def is_globex_code(symbol: str) -> bool:
		match = GlobexCode.GLOBEX_CODE_PATTERN.match(symbol)
		return match is not None

	def add_year(self) -> None:
		self.symbol = None
		self.year += 1

	def _get_tuple(self) -> tuple[str, int, str]:
		return self.root, self.year, self.month

	def __eq__(self, other: GlobexCode) -> bool:
		return self._get_tuple() == other._get_tuple()

	def __lt__(self, other: GlobexCode) -> bool:
		return self._get_tuple() < other._get_tuple() # type: ignore

	def __hash__(self) -> int:
		globex_tuple = self._get_tuple()
		output = hash(globex_tuple)
		return output

	def __repr__(self) -> str:
		if self.symbol is not None:
			return self.symbol
		else:
			return f"{self.root}{self.month}{self.year % 100:02}"

	def __copy__(self):
		return GlobexCode(self.symbol, self.root, self.month, self.year)