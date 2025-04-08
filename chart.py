import os

import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import seaborn as sns
import pandas as pd

from configuration import Configuration

def render_chart(symbols: list[str], start: pd.Timestamp | None, end: pd.Timestamp | None):
	fig, ax = plt.subplots(figsize=(12, 8))
	xlim_min_values = []
	xlim_max_values = []
	for symbol in symbols:
		path = os.path.join(Configuration.FEATHER_DIRECTORY, f"{symbol}.feather")
		df = pd.read_feather(path, columns=["time", "close"])
		df["time"] = pd.to_datetime(df["time"])
		# Apply filters in one go for performance reasons
		if start is not None and end is not None:
			df = df[(df["time"] >= start) & (df["time"] < end)]
		elif start is not None:
			df = df[(df["time"] >= start)]
		elif end is not None:
			df = df[(df["time"] < end)]
		xlim_min_values.append(df["time"].min())
		xlim_max_values.append(df["time"].max())
		sns.lineplot(ax=ax, x=df["time"], y=df["close"], label=symbol)
	xlim_min = min(xlim_min_values)
	xlim_max = max(xlim_max_values)
	plt.xlim(xlim_min, xlim_max)

	plt.ion()
	_cursor = Cursor(ax, horizOn=False, vertOn=True, color="grey", linewidth=1, alpha=0.2)

	plt.xlabel("Time")
	plt.ylabel("Price")
	plt.legend()
	plt.tight_layout()
	plt.show(block=True)
	plt.close()