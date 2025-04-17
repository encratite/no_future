import pandas as pd

class BacktestConfiguration:
	start: pd.Timestamp
	end: pd.Timestamp
	initial_cash: float

	def __init__(
		self,
		start: pd.Timestamp,
		end: pd.Timestamp,
		initial_cash: float
	):
		assert start < end
		assert initial_cash >= 10000
		self.start = start
		self.end = end
		self.initial_cash = initial_cash