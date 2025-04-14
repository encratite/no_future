from itertools import product
from typing import Any

import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoCV, ARDRegression, BayesianRidge

from common import has_free_threading
from configuration import Configuration
from enums import ModelType
from wrapper import PyTorchWrapper

def get_models(feature_count: int) -> list[tuple[str, ModelType, Any, dict]]:
	models = get_linear_models()
	models += get_random_forest_models()
	models += get_lightgbm_models()
	if Configuration.ENABLE_PYTORCH_MODELS:
		models += get_pytorch_models(feature_count)
	return models

def get_linear_models() -> list[tuple[str, ModelType, Any, dict]]:
	return [
		("LinearRegression", ModelType.LINEAR_REGRESSION, LinearRegression(), {}),
		("LassoCV", ModelType.LASSO_CV, LassoCV(max_iter=10000, random_state=Configuration.SEED), {}),
		("ElasticNetCV", ModelType.ELASTICNET_CV, ElasticNetCV(max_iter=10000, random_state=Configuration.SEED), {}),
		("ARDRegression", ModelType.ARD_REGRESSION, ARDRegression(), {}),
		("BayesianRidge", ModelType.BAYESIAN_RIDGE, BayesianRidge(), {}),
	]

def get_random_forest_models() -> list[tuple[str, ModelType, Any, dict]]:
	n_estimators_values = [
		# 25,
		# 50,
		# 75,
		# 100,
		125,
		# 150,
		# 200
	]
	criterion_values = [
		"squared_error",
		# "absolute_error",
		# "friedman_mse"
	]
	max_depths_values = [
		# None,
		2,
		# 3,
		# 4,
		# 5,
		# 6,
		# 7,
	]
	models = []
	combinations = product(
		n_estimators_values,
		criterion_values,
		max_depths_values
	)
	for n_estimators, criterion, max_depth in combinations:
		n_jobs = -1 if has_free_threading() else 1
		model = RandomForestRegressor(
			n_estimators=n_estimators,
			criterion=criterion,
			max_depth=max_depth,
			random_state=Configuration.SEED,
			n_jobs=n_jobs
		)
		parameters = {
			"n_estimators": n_estimators,
			"criterion": criterion,
			"max_depth": max_depth
		}
		models.append(("RandomForestRegressor", ModelType.RANDOM_FOREST, model, parameters))
	return models

def get_lightgbm_models() -> list[tuple[str, ModelType, Any, dict]]:
	num_leaves_values = [
		# 4,
		5,
		# 6,
		# 10,
		# 15,
		# 20,
		# 30,
		# 40,
		# 50
	]
	min_data_in_leaf_values = [
		0,
		1,
		2,
		# 3,
		# 5,
		# 10,
		# 15,
		# 20,
		# 30,
		# 40,
	]
	max_depth_values = [
		# -1,
		2,
		# 3,
		# 4,
		# 5,
		# 6,
		# 7,
	]
	num_iterations_values = [
		15,
		20,
		25,
		30,
		40,
		# 50,
		# 75,
		# 100
	]
	learning_rate_values = [
		0.01,
	]
	models = []
	combinations = product(
		num_leaves_values,
		min_data_in_leaf_values,
		max_depth_values,
		num_iterations_values,
		learning_rate_values
	)
	for num_leaves, min_data_in_leaf, max_depth, num_iterations, learning_rate in combinations:
		parameters = {
			"num_leaves": num_leaves,
			"min_data_in_leaf": min_data_in_leaf,
			"max_depth": max_depth,
			"num_iterations": num_iterations,
			# "learning_rate": learning_rate,
		}
		model = lgb.LGBMRegressor(
			num_leaves=num_leaves,
			min_data_in_leaf=min_data_in_leaf,
			max_depth=max_depth,
			num_iterations=num_iterations,
			learning_rate=learning_rate,
			verbosity=-1,
			seed=Configuration.SEED
		)
		models.append(("LGBMRegressor", ModelType.LIGHTGBM, model, parameters))
	return models

def get_pytorch_models(feature_count: int) -> list[tuple[str, ModelType, Any, dict]]:
	hidden_values = [
		# (8, 4),
		# (8, 4, 2),
		# (16, 8),
		(16, 8, 4),
		# (20, 10, 5),
		# (32, 16),
		# (32, 16, 8),
		# (64, 32)
	]
	activation_values = [
		"relu",
		# "sigmoid",
		# "tanh"
	]
	optimizer_values = [
		# "adam",
		"sgd"
	]
	batch_size_values = [
		1,
		# 4,
		# 8,
		# 16,
		# 32
	]
	learning_rate_values = [
		0.01,
		# 0.05,
		# 0.1
	]
	momentum_values = [
		0,
		# 0.5,
		# 0.9,
		# 0.99
	]
	epochs = [
		# 10,
		25,
		30,
		35,
		40,
		45,
		50,
		# 50,
		# 60,
		# 100
	]
	combinations = product(
		hidden_values,
		activation_values,
		optimizer_values,
		batch_size_values,
		learning_rate_values,
		momentum_values,
		epochs
	)
	models = []
	for hidden, activation, optimizer, batch_size, learning_rate, momentum, epochs in combinations:
		parameters = {
			"hidden": hidden,
			"activation": activation,
			"optimizer": optimizer,
			"batch_size": batch_size,
			"learning_rate": learning_rate,
			"momentum": momentum,
			"epochs": epochs
		}
		model = PyTorchWrapper(
			feature_count,
			hidden,
			activation,
			optimizer,
			batch_size,
			learning_rate,
			momentum,
			epochs
		)
		models.append(("PyTorch", ModelType.PYTORCH, model, parameters))
	return models