import pandas as pd

class BacktestConfiguration:
	start: pd.Timestamp
	end: pd.Timestamp
	initial_cash: float
	enable_output: bool

	def __init__(
		self,
		start: pd.Timestamp,
		end: pd.Timestamp,
		initial_cash: float,
		enable_output: bool = False
	):
		assert start < end
		assert initial_cash >= 10000
		self.start = start
		self.end = end
		self.initial_cash = initial_cash
		self.enable_output = enable_output