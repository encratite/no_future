class Asset:
	symbol: str
	name: str
	currency: str
	tick_size: float
	tick_value: float
	margin: float
	overnight_margin: bool
	margin_close: float | None
	broker_fee: float
	exchange_fee: float
	spread: int

	def __init__(
		self,
		symbol: str,
		name: str,
		currency: str,
		tick_size: float,
		tick_value: float,
		margin: float,
		overnight_margin: bool,
		broker_fee: float,
		exchange_fee: float,
		spread: int
	):
		self.symbol = symbol
		self.name = name
		self.currency = currency
		self.tick_size = tick_size
		self.tick_value = tick_value
		self.margin = margin
		self.overnight_margin = overnight_margin
		self.margin_close = None
		self.broker_fee = broker_fee
		self.exchange_fee = exchange_fee
		self.spread = spread