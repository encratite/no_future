from typing import Any

from enums import ModelType

class RegressionResult:
	symbol: str
	model_name: str
	model_type: ModelType
	parameters: dict[str, Any]
	r2_score_training: float
	r2_score_validation: float
	feature_count: int

	def __init__(
		self,
		symbol: str,
		model_name: str,
		model_type: ModelType,
		parameters: dict[str, Any],
		r2_score_training: float,
		r2_score_validation: float,
		feature_count: int
	) -> None:
		self.symbol = symbol
		self.model_name = model_name
		self.model_type = model_type
		self.parameters = parameters
		self.r2_score_training = r2_score_training
		self.r2_score_validation = r2_score_validation
		self.feature_count = feature_count