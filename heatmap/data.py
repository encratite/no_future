from typing import Any

from .constant import VOLATILITY_DAYS

class HeatmapFeature:
	id_: str
	description: str
	values: list[float]

	def __init__(self, id_: str, description: str):
		self.id_ = id_
		self.description = description
		self.values = []

	def append(self, value: float) -> None:
		self.values.append(value)

class HeatmapData:
	y_values: list[float]
	momentum2: HeatmapFeature
	momentum3: HeatmapFeature
	momentum10: HeatmapFeature
	regime: HeatmapFeature
	volume: HeatmapFeature
	open_interest: HeatmapFeature
	volatility: HeatmapFeature

	def __init__(self):
		self.y_values = []
		self.momentum2 = HeatmapFeature("momentum2", "Momentum (2 Days)")
		self.momentum3 = HeatmapFeature("momentum3", "Momentum (3 Days)")
		self.momentum10 = HeatmapFeature("momentum10", f"Momentum (10 Days)")
		self.regime = HeatmapFeature("regime", "Regime")
		self.volume = HeatmapFeature("volume", "Change in volume from yesterday")
		self.open_interest = HeatmapFeature("interest", "Change in open interest from yesterday")
		self.volatility = HeatmapFeature("volatility", f"Volatility ({VOLATILITY_DAYS} Days)")

	def get_values(self) -> dict[str, tuple[str, Any]]:
		output = {}
		features = [
			self.momentum2,
			self.momentum3,
			self.momentum10,
			self.regime,
			self.volume,
			self.open_interest,
			self.volatility
		]
		for feature in features:
			quantile_values = get_quantile_transform(feature.values)
			output[feature.id_] = (feature.description, quantile_values)
		return output