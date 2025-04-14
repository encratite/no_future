from abc import ABC, abstractmethod
from statistics import mean
from typing import Final

import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score as get_r2_score

from common import format_percentage
from configuration import Configuration

class RegressionWrapper(ABC):
	@abstractmethod
	def fit(self, x: npt.NDArray, y: npt.NDArray) -> None:
		pass

	@abstractmethod
	def predict(self, x: npt.NDArray) -> npt.NDArray:
		pass

class PyTorchWrapper(RegressionWrapper):
	DEVICE: Final[str] = "cpu"

	_model: nn.Sequential
	_batch_size: int
	_learning_rate: float
	_epochs: int

	def __init__(
		self,
		features: int,
		hidden: tuple,
		activation: str,
		batch_size: int,
		learning_rate: float,
		epochs: int
	) -> None:
		activation_functions = {
			"relu": nn.ReLU,
			"sigmoid": nn.Sigmoid,
			"tanh": nn.Tanh
		}
		activation_function = activation_functions[activation]
		if len(hidden) == 2:
			hidden1, hidden2 = hidden
			self._model = nn.Sequential(
				nn.Linear(features, hidden1),
				activation_function(),
				nn.Linear(hidden1, hidden2),
				activation_function(),
				nn.Linear(hidden2, 1)
			)
		elif len(hidden) == 3:
			hidden1, hidden2, hidden3 = hidden
			self._model = nn.Sequential(
				nn.Linear(features, hidden1),
				activation_function(),
				nn.Linear(hidden1, hidden2),
				activation_function(),
				nn.Linear(hidden2, hidden3),
				activation_function(),
				nn.Linear(hidden3, 1)
			)
		else:
			raise Exception("Invalid neural network shape")
		self._model.to(self.DEVICE)
		self._batch_size = batch_size
		self._learning_rate = learning_rate
		self._epochs = epochs

	def fit(self, x: npt.NDArray, y: npt.NDArray) -> None:
		early_stopping_samples = 4 * Configuration.MAX_BATCH_SIZE
		x_training = x[:-early_stopping_samples]
		y_training = y[:-early_stopping_samples]
		x_validation = x[-early_stopping_samples:]
		y_validation = y[-early_stopping_samples:]
		x_training_tensor = torch.tensor(x_training, dtype=torch.float32, device=self.DEVICE)
		y_training_tensor = torch.tensor(y_training, dtype=torch.float32, device=self.DEVICE).unsqueeze(1)
		x_validation_tensor = torch.tensor(x_validation, dtype=torch.float32, device=self.DEVICE)
		dataset = TensorDataset(x_training_tensor, y_training_tensor)
		generator = torch.Generator()
		generator.manual_seed(Configuration.SEED)
		data_loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True, generator=generator)
		criterion = nn.MSELoss()
		optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)
		r2_score_buffer_size: Final[int] = 10
		previous_r2_scores: list[float] = []
		for epoch in range(self._epochs):
			self._model.train()
			for batch_x, batch_y in data_loader:
				optimizer.zero_grad()
				outputs = self._model(batch_x)
				loss = criterion(outputs, batch_y)
				loss.backward()
				optimizer.step()

			self._model.eval()
			with torch.no_grad():
				training_predictions = self._model(x_training_tensor)
				validation_predictions = self._model(x_validation_tensor)
				training_predictions_np = training_predictions.squeeze().cpu().numpy()
				validation_predictions_np = validation_predictions.squeeze().cpu().numpy()
				r2_score_training = get_r2_score(y_training, training_predictions_np)
				r2_score_validation = get_r2_score(y_validation, validation_predictions_np)
				print(f"Epoch {epoch + 1}/{self._epochs}: IS R^2 {format_percentage(r2_score_training)}, OOS R^2 {format_percentage(r2_score_validation)}")
				previous_r2_scores.append(r2_score_validation)
				if len(previous_r2_scores) > r2_score_buffer_size:
					previous_r2_scores = previous_r2_scores[1:]
				if len(previous_r2_scores) >= r2_score_buffer_size and r2_score_training > 0 and r2_score_validation < mean(previous_r2_scores):
					print("Detected OOS model degradation, stopping training")
					break

	def predict(self, x: npt.NDArray) -> npt.NDArray:
		x_tensor = torch.tensor(x, dtype=torch.float32, device=self.DEVICE)
		self._model.eval()
		with torch.no_grad():
			predictions = self._model(x_tensor)
			predictions_np = predictions.squeeze().cpu().numpy()
			return predictions_np