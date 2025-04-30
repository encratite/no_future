from enum import Enum
from typing import Final

from backtest.interface import BacktestInterface
from .base import Strategy

class NewHighLowMode(Enum):
	LONG_ON_HIGH_CLOSE: Final[int] = 0
	LONG_ON_HIGH: Final[int] = 1
	LONG_ON_LOW_CLOSE: Final[int] = 2
	LONG_ON_LOW: Final[int] = 3
	SHORT_ON_HIGH_CLOSE: Final[int] = 4
	SHORT_ON_HIGH: Final[int] = 5
	SHORT_ON_LOW_CLOSE: Final[int] = 6
	SHORT_ON_LOW: Final[int] = 7

class NewHighLowStrategy(Strategy):
	_symbol: str
	_window_size: int
	_holding_time: int
	_mode: NewHighLowMode
	_remaining_holding_time: int

	def __init__(
		self,
		symbol: str,
		window_size: int,
		holding_time: int,
		mode: NewHighLowMode
	) -> None:
		assert window_size > 0
		assert holding_time > 0
		mode_strings = {
			NewHighLowMode.LONG_ON_HIGH_CLOSE: "long on high (close)",
			NewHighLowMode.LONG_ON_HIGH: "long on high",
			NewHighLowMode.LONG_ON_LOW_CLOSE: "long on low (close)",
			NewHighLowMode.LONG_ON_LOW: "long on low",
			NewHighLowMode.SHORT_ON_HIGH_CLOSE: "short on high (close)",
			NewHighLowMode.SHORT_ON_HIGH: "short on high",
			NewHighLowMode.SHORT_ON_LOW_CLOSE: "short on low (close)",
			NewHighLowMode.SHORT_ON_LOW: "short on low"
		}
		mode_string = mode_strings[mode]
		super().__init__(f"New High/Low ({window_size} day window, {holding_time} day holding time, {mode_string})")
		self._symbol = symbol
		self._window_size = window_size
		self._holding_time = holding_time
		self._mode = mode

	def get_signals(self, interface: BacktestInterface) -> dict[str, float]:
		output = {}
		if self._remaining_holding_time == 0:
			records = interface.get_records(self._symbol, count=self._window_size)
			today = records[0]
			closes = [x.close for x in records]
			highs = [x.high for x in records]
			lows = [x.low for x in records]
			if today.time == interface.time:
				conditions = {
					NewHighLowMode.LONG_ON_HIGH_CLOSE: today.close == max(closes),
					NewHighLowMode.LONG_ON_HIGH: today.high == max(highs),
					NewHighLowMode.LONG_ON_LOW_CLOSE: today.low == min(closes),
					NewHighLowMode.LONG_ON_LOW: today.low == min(lows),
					NewHighLowMode.SHORT_ON_HIGH_CLOSE: today.close == max(closes),
					NewHighLowMode.SHORT_ON_HIGH: today.high == max(highs),
					NewHighLowMode.SHORT_ON_LOW_CLOSE: today.low == min(closes),
					NewHighLowMode.SHORT_ON_LOW: today.low == min(lows),
				}
				condition = conditions[self._mode]
				if condition:
					output = self._get_signal()
					self._remaining_holding_time = self._holding_time - 1
		else:
			output = self._get_signal()
			self._remaining_holding_time -= 1
		return output

	def reset(self) -> None:
		self._remaining_holding_time = 0

	def _get_signal(self) -> dict[str, float]:
		signals = {
			NewHighLowMode.LONG_ON_HIGH_CLOSE: 1,
			NewHighLowMode.LONG_ON_HIGH: 1,
			NewHighLowMode.LONG_ON_LOW_CLOSE: 1,
			NewHighLowMode.LONG_ON_LOW: 1,
			NewHighLowMode.SHORT_ON_HIGH_CLOSE: -1,
			NewHighLowMode.SHORT_ON_HIGH: -1,
			NewHighLowMode.SHORT_ON_LOW_CLOSE: -1,
			NewHighLowMode.SHORT_ON_LOW: -1,
		}
		signal = signals[self._mode]
		output = {
			self._symbol: signal
		}
		return output