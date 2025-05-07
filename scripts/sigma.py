from statistics import mean, stdev

import datetime as dt

from common import read_ohlc_series, get_rate_of_change

def evaluate_sigma() -> None:
	series = read_ohlc_series("ES", True)
	records = series.values()
	all_returns = []
	for previous_record, record in zip(records, records[1:]):
		if record.time.time() == dt.time(17, 0):
			returns = get_rate_of_change(record.close, previous_record.close)
			all_returns.append(returns)
	returns_mean = mean(all_returns)
	returns_sigma = stdev(all_returns)
	sample = get_rate_of_change(5675.0, 5615.0)
	z_score = (sample - returns_mean) / returns_sigma
	print(f"Mean: {returns_mean:.3%}")
	print(f"Standard deviation: {returns_sigma:.3%}")
	print(f"Z-score of sample ({sample:.2%}): {z_score:.2f}")

evaluate_sigma()