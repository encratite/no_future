import os
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import get_rate_of_change
from configuration import Configuration
from series import TimeSeries, OhlcRecord

CUTOFF_DATE = pd.Timestamp("2008-01-01")
BUFFER_SIZE = 700
STRIDE = 1
COLUMN = "Quantile"

class Sample:
	momentum: float
	rank: int

	def __init__(self, momentum: float, rank: int) -> None:
		self.momentum = momentum
		self.rank = rank

def quantile_transform(symbol: str) -> None:
	path = os.path.join(Configuration.BARCHART_DIRECTORY, f"{symbol}.csv")
	series = TimeSeries.read_ohlc_csv(path)
	records = series.values()
	momentum_values = []
	for record1, record2 in zip(records[1:], records):
		record1 = cast(OhlcRecord, record1)
		record2 = cast(OhlcRecord, record2)
		if record1.time < CUTOFF_DATE or record2.time < CUTOFF_DATE:
			continue
		momentum = get_rate_of_change(record1.close, record2.close)
		momentum_values.append(momentum)
	quantiles = generate_quantiles(momentum_values[:BUFFER_SIZE], BUFFER_SIZE)
	offset = BUFFER_SIZE
	while offset + BUFFER_SIZE < len(momentum_values):
		truncated_values = momentum_values[offset:offset + BUFFER_SIZE]
		quantiles += generate_quantiles(truncated_values, STRIDE)
		offset += STRIDE
	df = pd.DataFrame({
		COLUMN: quantiles
	})
	sns.histplot(data=df, x=COLUMN, stat="density", bins=50)
	plt.show()

def generate_quantiles(values: list[float], quantile_samples: int) -> list[float]:
	samples = []
	for i, x in enumerate(values):
		sample = Sample(x, i)
		samples.append(sample)
	samples = sorted(samples, key=lambda s: s.momentum)
	quantiles = []
	for i, x in enumerate(samples):
		if x.rank >= len(values) - quantile_samples:
			quantile = i / (len(values) - 1)
			quantiles.append(quantile)
	assert(len(quantiles) == quantile_samples)
	return quantiles

quantile_transform("SPY.D1")