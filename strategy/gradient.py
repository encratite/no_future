from statistics import mean

from backtest.interface import BacktestInterface
from constant import TRADING_DAYS_PER_YEAR
from .base import Strategy

class GradientStrategy(Strategy):
	_symbol: str
	_window_size: int
	_signal_count: int
	_regime_filter: bool
	_invert: bool

	_moving_average_buffer: list[float]

	def __init__(
		self,
		symbol: str,
		window_size: int,
		signal_count: int,
		regime_filter: bool,
		invert: bool
	) -> None:
		assert window_size > 0
		assert signal_count > 0
		descriptions = [
			symbol,
			f"{window_size} day window",
			f"{signal_count} successive signals",
		]
		if regime_filter:
			descriptions.append("regime filter")
		if invert:
			descriptions.append("invert signal")
		super().__init__(f"Gradient ({", ".join(descriptions)})")
		self._symbol = symbol
		self._window_size = window_size
		self._signal_count = signal_count
		self._regime_filter = regime_filter
		self._invert = invert
		self.reset()

	def get_signals(self, interface: BacktestInterface) -> dict[str, float]:
		if self._regime_filter:
			count = TRADING_DAYS_PER_YEAR
		else:
			count = self._window_size
		records = interface.get_records(self._symbol, count=count)
		closes = [x.close for x in records]
		moving_average_window = closes[:self._window_size]
		moving_average = mean(moving_average_window)
		self._moving_average_buffer = [moving_average] + self._moving_average_buffer
		limit = self._signal_count + 1
		length = len(self._moving_average_buffer)
		if length > limit:
			self._moving_average_buffer = self._moving_average_buffer[:-1]
		elif length < limit:
			return {}
		deltas = [a - b for a, b in zip(self._moving_average_buffer, self._moving_average_buffer[1:])]
		assert len(deltas) == self._signal_count
		long = all(x > 0 for x in deltas)
		short = all(x < 0 for x in deltas)
		if long:
			signal = 1
		elif short:
			signal = -1
		else:
			signal = 0
		if self._invert:
			signal = - signal
		if self._regime_filter:
			if records[0].close > records[-1].close:
				regime_signal = 1
			else:
				regime_signal = -1
			if signal != regime_signal:
				signal = 0
		output = {
			self._symbol: signal
		}
		return output

	def reset(self) -> None:
		self._moving_average_buffer = []