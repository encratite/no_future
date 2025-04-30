from enum import Enum
from statistics import mean
from typing import Final

from backtest.interface import BacktestInterface
from ohlc import OhlcRecord
from .base import Strategy

class MovingAverageTradingMode(Enum):
	LONG: Final[int] = 0
	SHORT: Final[int] = 1
	LONG_SHORT: Final[int] = 2

class MovingAverageFunction(Enum):
	SIMPLE: Final[int] = 0
	EXPONENTIAL: Final[int] = 1

class MovingAverageConfiguration:
	fast_days: int
	slow_days: int
	trading_mode: MovingAverageTradingMode
	function: MovingAverageFunction
	regime_filter: bool
	holding_time: int | None

	def __init__(
		self,
		fast_days: int,
		slow_days: int,
		trading_mode: MovingAverageTradingMode,
		function: MovingAverageFunction,
		regime_filter: bool,
		holding_time: int | None
	):
		assert 2 <= fast_days < slow_days
		assert holding_time >= 1
		self.fast_days = fast_days
		self.slow_days = slow_days
		self.trading_mode = trading_mode
		self.function = function
		self.regime_filter = regime_filter
		self.holding_time = holding_time

class MovingAverageStrategy(Strategy):
	_symbol: str
	_configuration: MovingAverageConfiguration
	_previous_signal: int
	_remaining_holding_time: int | None

	def __init__(
		self,
		symbol: str,
		configuration: MovingAverageConfiguration
	):
		trading_mode_strings = {
			MovingAverageTradingMode.LONG: "long only",
			MovingAverageTradingMode.SHORT: "short only",
			MovingAverageTradingMode.LONG_SHORT: "long/short",
		}
		trading_mode_description = trading_mode_strings[configuration.trading_mode]
		function_strings = {
			MovingAverageFunction.SIMPLE: "SMA",
			MovingAverageFunction.EXPONENTIAL: "EMA",
		}
		function_description = function_strings[configuration.function]
		if configuration.regime_filter:
			regime_filter_description = ", regime filter"
		else:
			regime_filter_description = ""
		if configuration.holding_time is not None:
			super().__init__(f"Moving Average Trigger ({configuration.fast_days} fast, {configuration.slow_days} slow, {trading_mode_description}, {function_description}, {configuration.holding_time} days{regime_filter_description})")
		else:
			super().__init__(f"Moving Average Crossover ({configuration.fast_days} fast, {configuration.slow_days} slow, {trading_mode_description}, {function_description}{regime_filter_description})")
		self._symbol = symbol
		self._configuration = configuration
		self.reset()

	def get_signals(self, interface: BacktestInterface) -> dict[str, float]:
		records_count = 250 if self._configuration.regime_filter else self._configuration.slow_days
		all_records = interface.get_records(self._symbol, count=records_count)
		fast_records = all_records[:self._configuration.fast_days]
		slow_records = all_records[:self._configuration.slow_days]
		functions = {
			MovingAverageFunction.SIMPLE: self._get_simple_moving_average,
			MovingAverageFunction.EXPONENTIAL: self._get_exponential_moving_average,
		}
		function = functions[self._configuration.function]
		fast_moving_average = function(fast_records)
		slow_moving_average = function(slow_records)
		signal = 1 if fast_moving_average >= slow_moving_average else -1
		if self._configuration.regime_filter:
			closes = [x.close for x in all_records]
			today = closes[0]
			simple_moving_average = mean(closes)
			if signal == 1 and today < simple_moving_average or signal == -1 and today > simple_moving_average:
				return {}
		output = {}
		if self._configuration.holding_time is not None:
			if self._remaining_holding_time is None:
				if self._previous_signal != signal:
					self._remaining_holding_time = self._configuration.holding_time - 1
					if self._signal_match(signal):
						output[self._symbol] = signal
				self._previous_signal = signal
			else:
				if self._remaining_holding_time > 0:
					self._remaining_holding_time -= 1
					output[self._symbol] = self._previous_signal
				else:
					self._remaining_holding_time = None
					self._previous_signal = signal
		else:
			if self._signal_match(signal):
				output[self._symbol] = signal
		return output

	def reset(self) -> None:
		self._previous_signal = 0
		self._remaining_holding_time = None

	@staticmethod
	def _get_simple_moving_average(records: list[OhlcRecord]) -> float:
		closes = [x.close for x in records]
		average = mean(closes)
		return average

	@staticmethod
	def _get_exponential_moving_average(records: list[OhlcRecord]) -> float:
		sum_ = 0
		coefficient_sum = 0
		i = 0
		lambda_ = 2.0 / (len(records) + 1)
		for x in records:
			coefficient = lambda_ * (1 - lambda_)**i
			sum_ += coefficient * x.close
			coefficient_sum += coefficient
			i += 1
		average = sum_ / coefficient_sum
		return average

	def _signal_match(self, signal: int) -> bool:
		long_enums = [MovingAverageTradingMode.LONG, MovingAverageTradingMode.LONG_SHORT]
		short_enums = [MovingAverageTradingMode.SHORT, MovingAverageTradingMode.LONG_SHORT]
		long_match = self._configuration.trading_mode in long_enums and signal == 1
		short_match = self._configuration.trading_mode in short_enums and signal == -1
		match = long_match or short_match
		return match