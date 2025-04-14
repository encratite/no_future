from typing import Final
from enum import Enum

class ModelType(Enum):
	LINEAR_REGRESSION: Final[int] = 0
	LASSO_CV: Final[int] = 1
	ELASTICNET_CV: Final[int] = 2
	ARD_REGRESSION: Final[int] = 3
	BAYESIAN_RIDGE: Final[int] = 4
	RANDOM_FOREST: Final[int] = 5
	LIGHTGBM: Final[int] = 6
	PYTORCH: Final[int] = 7