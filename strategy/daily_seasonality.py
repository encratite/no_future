from collections import defaultdict
from statistics import mean

import numpy as np
import pandas as pd

from backtest.interface import BacktestInterface
from .base import Strategy

class DailySeasonalityData:
	best_day: int
	mean_return: float
	signal_sign: int
	last_update: pd.Timestamp

	def __init__(
		self,
		best_day: int,
		mean_return: float,
		signal_sign: int,
		last_update: pd.Timestamp
	):
		self.best_day = best_day
		self.mean_return = mean_return
		self.signal_sign = signal_sign
		self.last_update = last_update

class DailySeasonalityStrategy(Strategy):
	_sample_size: int
	_minimum_return: float
	_symbols: list[str]
	_target_notional_value: float
	_best_days: dict[str, DailySeasonalityData] | None

	def __init__(
		self,
		sample_size: int,
		minimum_return: float,
		symbols: list[str],
		target_notional_value: float
	):
		super().__init__(f"Daily Seasonality ({sample_size} Samples, Minimum {minimum_return})")
		self._sample_size = sample_size
		self._minimum_return = minimum_return
		self._symbols = symbols
		self._target_notional_value = target_notional_value
		self._best_days = {}

	def get_signals(self, interface: BacktestInterface) -> dict[str, float]:
		signals: dict[str, float] = {}
		for symbol in self._symbols:
			if self.is_banned_symbol(symbol, interface.time):
				continue
			if symbol not in self._best_days:
				self._calculate_best_days(symbol, interface)
			if symbol not in self._best_days:
				continue
			data = self._best_days[symbol]
			time_delta = interface.time - data.last_update
			if time_delta.days >= 20:
				self._calculate_best_days(symbol, interface)
			data = self._best_days[symbol]
			if interface.time.day_of_week == data.best_day and abs(data.mean_return) > self._minimum_return:
				today = interface.get_record(symbol)
				fractional_contracts = interface.get_contracts(symbol, today, self._target_notional_value)
				signal = data.signal_sign * fractional_contracts
				signals[symbol] = signal
		return signals

	def _calculate_best_days(self, symbol: str, interface: BacktestInterface) -> None:
		records = interface.get_records(symbol, interface.time, self._sample_size)
		records = [x for x in records if not self.is_banned_symbol(symbol, x.time)]
		if len(records) < self._sample_size / 2.0:
			if symbol in self._best_days:
				self._best_days[symbol].last_update = interface.time
			return
		daily_returns: list[tuple[pd.Timestamp, float]] = [(a.time, b.close / a.close - 1) for a, b in zip(records, records[1:])]
		returns_by_day: defaultdict[int, list[float]] = defaultdict(list)
		for time, returns in daily_returns:
			returns_by_day[time.day_of_week].append(returns)
		mean_returns_by_day: dict[int, float] = {}
		for day_of_week, returns_list in returns_by_day.items():
			mean_returns_by_day[day_of_week] = mean(returns_list)
		best_tuple: tuple[int, float] = max(mean_returns_by_day.items(), key=lambda x: abs(x[1]))
		best_day, best_returns = best_tuple
		signal_sign = np.sign(best_returns)
		data = DailySeasonalityData(best_day, best_returns, signal_sign, interface.time)
		self._best_days[symbol] = data