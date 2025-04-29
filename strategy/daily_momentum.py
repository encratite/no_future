import calendar
from collections import defaultdict, deque
from statistics import mean, stdev
from typing import Any

import pandas as pd

from backtest.interface import BacktestInterface
from common import get_rate_of_change
from constant import TRADING_DAYS_PER_WEEK
from .base import Strategy

class SymbolDay:
	symbol: str
	day: int

	def __init__(self, symbol: str, day: int) -> None:
		self.symbol = symbol
		self.day = day

	def __eq__(self, other: Any) -> bool:
		if isinstance(other, SymbolDay):
			return (self.symbol, self.day) == (other.symbol, other.day)
		else:
			return False

	def __hash__(self) -> int:
		return hash((self.symbol, self.day))

	def __repr__(self) -> str:
		return f"{self.symbol} ({calendar.day_name[self.day]})"

class DailyMomentumStrategy(Strategy):
	_symbols: list[str]
	_long_only_symbols: list[str]
	_weeks: int
	_long_count: int
	_short_count: int
	_buffer: defaultdict[tuple[str, int], deque[float]] | None
	_last_execution: pd.Timestamp | None
	_signals: dict[SymbolDay, int] | None

	def __init__(
		self,
		symbols: list[str],
		long_only_symbols: list[str],
		weeks: int = 8,
		long_count: int = 1,
		short_count: int = 1
	) -> None:
		assert len(symbols) > 0 and weeks > 0
		symbol_string = ", ".join(symbols)
		super().__init__(f"Daily Momentum ({symbol_string}, {weeks}, {long_count}, {short_count})")
		self._symbols = symbols
		self._long_only_symbols = long_only_symbols
		self._weeks = weeks
		self._long_count = long_count
		self._short_count = short_count
		self.reset()

	def get_signals(self, interface: BacktestInterface) -> dict[str, float]:
		if self._buffer is None:
			self._initialize_buffer(interface)
		else:
			self._update_buffer(interface)
		if self._last_execution is None or self._last_execution.week != interface.time.week:
			self._select_assets(interface)
		signals: dict[str, float] = {}
		if self._signals is not None:
			for key, signal in self._signals.items():
				if interface.time.dayofweek == key.day:
					signals[key.symbol] = signal
		return signals

	def reset(self) -> None:
		self._buffer = None
		self._last_execution = None
		self._signals = None

	def _initialize_buffer(self, interface: BacktestInterface) -> None:
		self._buffer = defaultdict(deque)
		for symbol in self._symbols:
			days = 2 * self._weeks * TRADING_DAYS_PER_WEEK
			records = interface.get_records(symbol, count=days)
			time_returns = [(b.time, get_rate_of_change(a.close, b.close)) for a, b in zip(records, records[1:])]
			for time, returns in time_returns:
				key = SymbolDay(symbol, time.dayofweek)
				self._buffer[key].appendleft(returns)
		self._truncate_buffer()

	def _update_buffer(self, interface: BacktestInterface) -> None:
		for symbol in self._symbols:
			records = interface.get_records(symbol, count=2)
			if records[0].time != interface.time:
				continue
			key = SymbolDay(symbol, records[1].time.dayofweek)
			returns = get_rate_of_change(records[0].close, records[1].close)
			self._buffer[key].appendleft(returns)
		self._truncate_buffer()

	def _truncate_buffer(self) -> None:
		for returns in self._buffer.values():
			while len(returns) > self._weeks:
				returns.pop()

	def _select_assets(self, interface: BacktestInterface) -> None:
		ratings: list[tuple[SymbolDay, float]] = []
		for key, returns in self._buffer.items():
			if len(returns) >= 2:
				risk_adjusted_return = mean(returns) / stdev(returns)
			else:
				risk_adjusted_return = mean(returns)
			ratings.append((key, risk_adjusted_return))
		sorted_ratings = sorted(ratings, key=lambda x: x[1])
		long_targets = [x for x in sorted_ratings if x[1] > 0]
		short_targets = [x for x in sorted_ratings if x[1] < 0 and x[0].symbol not in self._long_only_symbols]
		long_assets = long_targets[-self._long_count:]
		short_assets = short_targets[:self._short_count]
		self._signals = {}
		for key, _rating in long_assets:
			self._signals[key] = 1
		for key, _rating in short_assets:
			self._signals[key] = -1
		print(f"{interface.time} Selected assets: {self._signals}")
		self._last_execution = interface.time