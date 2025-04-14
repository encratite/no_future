from abc import ABC, abstractmethod
from typing import Final

import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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
		x_tensor = torch.tensor(x, dtype=torch.float32, device=self.DEVICE)
		y_tensor = torch.tensor(y, dtype=torch.float32, device=self.DEVICE).unsqueeze(1)
		dataset = TensorDataset(x_tensor, y_tensor)
		generator = torch.Generator()
		generator.manual_seed(Configuration.SEED)
		data_loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True, generator=generator)
		criterion = nn.MSELoss()
		optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)
		for epoch in range(self._epochs):
			self._model.train()
			epoch_loss = 0
			for batch_x, batch_y in data_loader:
				optimizer.zero_grad()
				outputs = self._model(batch_x)
				loss = criterion(outputs, batch_y)
				loss.backward()
				optimizer.step()
				epoch_loss += loss.item() * batch_x.size(0)
			mean_loss = epoch_loss / len(dataset)
			print(f"Epoch {epoch + 1}/{self._epochs}: {mean_loss:.6f}")

	def predict(self, x: npt.NDArray) -> npt.NDArray:
		x_tensor = torch.tensor(x, dtype=torch.float32, device=self.DEVICE)
		self._model.eval()
		with torch.no_grad():
			predictions = self._model(x_tensor)
			predictions_np = predictions.squeeze().cpu().numpy()
			return predictions_np