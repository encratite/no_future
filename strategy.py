from abc import ABC, abstractmethod
from enum import Enum
from typing import Final

import pandas as pd
import numpy as np

from backtest_interface import BacktestInterface

class RebalanceMode(Enum):
	WEEKLY_MONDAY: Final[int] = 0
	WEEKLY_TUESDAY: Final[int] = 1
	WEEKLY_WEDNESDAY: Final[int] = 2
	WEEKLY_THURSDAY: Final[int] = 3
	WEEKLY_FRIDAY: Final[int] = 4
	START_OF_MONTH: Final[int] = 10
	END_OF_MONTH: Final[int] = 11

class Strategy(ABC):
	name: str
	weight: float

	def __init__(self, name: str):
		self.name = name
		self.weight = 1

	@abstractmethod
	def get_signals(self, interface: BacktestInterface) -> dict[str, float]:
		pass

	def reset(self) -> None:
		pass

class BuyAndHoldStrategy(Strategy):
	_signals: dict[str, float]

	def __init__(self, signals: dict[str, float]):
		super().__init__("Buy and Hold")
		self._signals = signals

	def get_signals(self, interface: BacktestInterface) -> dict[str, float]:
		return self._signals

class MomentumStrategy(Strategy):
	_months: int
	_rebalance_mode: RebalanceMode
	_symbols: list[str]
	_target_notional_value: float
	_previous_signals: dict[str, float]
	_rebalance_days: set[pd.Timestamp] | None

	def __init__(
		self,
		months: int,
		rebalance_mode: RebalanceMode,
		symbols: list[str],
		target_notional_value: float
	) -> None:
		assert 1 <= months <= 24
		months_string = "month" if months == 1 else "months"
		super().__init__(f"Momentum ({months} {months_string})")
		self._months = months
		self._rebalance_mode = rebalance_mode
		self._symbols = symbols
		self._target_notional_value = target_notional_value
		self.reset()

	def get_signals(self, interface: BacktestInterface) -> dict[str, float]:
		days_per_month: Final[float] = 365.25 / 12

		if self._rebalance_days is None:
			self._calculate_rebalance_days(interface)
		if interface.time not in self._rebalance_days:
			return self._previous_signals

		signals: dict[str, float] = {}
		for symbol in self._symbols:
			today = interface.get_record(symbol)
			days = self._months * days_per_month
			lookback_time = interface.time - pd.Timedelta(days=days)
			lookback = interface.get_record(symbol, lookback_time)
			if today.close <= 0 or lookback.close <= 0:
				signals[symbol] = 0
				continue
			momentum_sign = np.sign(today.close / lookback.close - 1)
			fractional_contracts = interface.get_contracts(symbol, today, self._target_notional_value)
			signal = momentum_sign * fractional_contracts
			signals[symbol] = signal
		self._previous_signals = signals
		return signals

	def reset(self) -> None:
		self._previous_signals = {}
		self._rebalance_days = None

	def _calculate_rebalance_days(self, interface: BacktestInterface) -> None:
		def new_rebalance_time() -> bool:
			if previous_rebalance_time is None:
				return True
			else:
				return previous_rebalance_time.month != time.month and interface.is_trading_day(time)

		self._rebalance_days = set()
		previous_rebalance_time: pd.Timestamp | None = None
		if self._rebalance_mode == RebalanceMode.START_OF_MONTH:
			time = interface.start
			while time < interface.end:
				if new_rebalance_time():
					self._rebalance_days.add(time)
					previous_rebalance_time = time
				time += pd.Timedelta(days=1)
		elif self._rebalance_mode == RebalanceMode.END_OF_MONTH:
			time = interface.end
			while time >= interface.start:
				if new_rebalance_time():
					self._rebalance_days.add(time)
					previous_rebalance_time = time
				time -= pd.Timedelta(days=1)
		else:
			time = interface.start
			while time < interface.end and time.day_of_week != 0:
				time += pd.Timedelta(days=1)
			while time < interface.end:
				week_time = time
				week_days = []
				while week_time.day_of_week <= 5:
					if interface.is_trading_day(week_time):
						week_days.append(week_time)
					week_time += pd.Timedelta(days=1)
				sorted_week_days = sorted(week_days, key=lambda x: abs(x.day_of_week - self._rebalance_mode.value))
				best_day = sorted_week_days[0]
				self._rebalance_days.add(best_day)
				time += pd.Timedelta(weeks=1)