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

	def __init__(
		self,
		fast_days: int,
		slow_days: int,
		trading_mode: MovingAverageTradingMode,
		function: MovingAverageFunction,
		regime_filter: bool
	):
		assert 2 <= fast_days < slow_days
		self.fast_days = fast_days
		self.slow_days = slow_days
		self.trading_mode = trading_mode
		self.function = function
		self.regime_filter = regime_filter

class MovingAverageStrategy(Strategy):
	_symbol: str
	_configuration: MovingAverageConfiguration

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
		super().__init__(f"Moving Average Crossover ({configuration.fast_days} fast, {configuration.slow_days} slow, {trading_mode_description}, {function_description}{regime_filter_description})")
		self._symbol = symbol
		self._configuration = configuration

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
		long_match = self._configuration.trading_mode in [MovingAverageTradingMode.LONG, MovingAverageTradingMode.LONG_SHORT] and signal == 1
		short_match = self._configuration.trading_mode in [MovingAverageTradingMode.SHORT, MovingAverageTradingMode.LONG_SHORT] and signal == -1
		if long_match or short_match:
			output[self._symbol] = signal
		return output

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