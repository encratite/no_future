from itertools import product
from typing import Any

import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoCV, ARDRegression, BayesianRidge

from common import has_free_threading
from configuration import Configuration
from enums import ModelType

def get_models() -> list[tuple[str, ModelType, Any, dict]]:
	models = get_linear_models()
	models += get_random_forest_models()
	models += get_lightgbm_models()
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
		3,
		4,
		5,
		# 6,
		# 7,
		# 8
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
		4,
		5,
		6,
		10,
		# 15,
		# 20,
		# 30,
		# 40,
		# 50
	]
	min_data_in_leaf_values = [
		# 5,
		10,
		15,
		20,
		30,
		40,
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
		20,
		25,
		30,
		40,
		50,
		# 60,
	]
	learning_rate_values = [
		0.03,
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