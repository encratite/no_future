from argparse import ArgumentParser
from typing import cast

import pandas as pd

from chart import render_chart
from chart import (
	render_comparison_chart,
	render_ratio_chart,
	render_seasonality_chart,
	SeasonalityChartMode
)
from generate import generate_all_contracts, generate_contract
from heatmap import render_heatmap, render_heatmap_all
from momentum import analyze_momentum
from seasonality import analyze_seasonality
from test.test_moving_average import perform_backtest
from test.test_wfo import perform_wfo_backtest

def get_date_argument(date_string: str) -> pd.Timestamp:
	pd.to_datetime(date_string, format="%Y-%m-%d", errors="raise")
	return pd.Timestamp(date_string)

def main() -> None:
	parser = ArgumentParser(description="Futures backtesting and analysis")
	group = parser.add_mutually_exclusive_group()
	group.add_argument("--generate-all", action="store_true", help="Generate continuous contracts for all symbols")
	group.add_argument("--generate", metavar="SYMBOL", help="Generate a continuous contract for the specified symbol")

	chart_help = "Render a chart for the specified symbols\n"
	chart_help += "By default this will display all available data\n"
	chart_help += "Use --start and --end to limit the chart to a certain time range"
	group.add_argument("--chart", metavar="SYMBOLS", nargs="*", help=chart_help)

	compare_help = "Compare the specified symbols, with all initial prices normalized to 1.0\n"
	compare_help += "Use --start and --end to limit the chart to a certain time range"
	group.add_argument("--compare", metavar="SYMBOLS", nargs="*", help=compare_help)

	chart_ratio_help = "Render a ratio chart using the specified symbols\n"
	chart_ratio_help += "Use --start and --end to limit the chart to a certain time range"
	group.add_argument("--chart-ratio", metavar=("BASE", "DIVISOR", "DIVIDEND"), nargs=3, help=chart_ratio_help)

	heatmap_help = "Render a heatmap of one-day returns of the specified symbol based on the quantiles of the two specified features\n"
	heatmap_help += "Supported features: momentum2, momentum3, momentum10, regime, gain2pain, volume, interest, volatility"
	heatmap_help += "The number of quantiles determines the number of cells in the heatmap"
	heatmap_help += "Requires the use of --start and --end"
	group.add_argument("--heatmap", metavar=("SYMBOL", "FEATURE1", "FEATURE2", "QUANTILES"), nargs=4, help=heatmap_help)

	heatmap_all_help = "Render all heatmaps for the specified symbol"
	heatmap_all_help += "Add --statistics to reduce output to just Welch's t-test"
	group.add_argument("--heatmap-all", metavar="SYMBOL", help=heatmap_all_help)
	parser.add_argument("--statistics", action="store_true", help="Do not render heatmaps, print tables only")

	seasonality_help = "Analyze seasonality of the specified symbols\n"
	seasonality_help += "Requires the use of --start, --split and --end"
	group.add_argument("--seasonality", metavar="SYMBOLS", nargs="*", help=seasonality_help)

	seasonality_chart_help = "Render a chart with cumulative seasonality time series for the specified symbol\n"
	seasonality_chart_help += "Supported modes: day, month, quarter\n"
	seasonality_chart_help += "Also requires the use of --start and --end"
	group.add_argument("--seasonality-chart", metavar=("SYMBOL", "MODE"), nargs=2, help=seasonality_chart_help)

	parser.add_argument("--start", metavar="DATE", type=get_date_argument, help="Restrict data to be read to after this date")
	parser.add_argument("--split", metavar="DATE", type=get_date_argument, help="Date at which to split in-sample and out-of-sample data")
	parser.add_argument("--end", metavar="DATE", type=get_date_argument, help="Read no data after this date")

	group.add_argument("--momentum", metavar="SYMBOLS", nargs="*", help="Analyze momentum correlation of a symbol")

	backtest_help = "Perform a backtest using the strategies defined in backtest_test.py\n"
	backtest_help += "Requires the use of --start and --end"
	group.add_argument("--backtest", action="store_true", help=backtest_help)

	backtest_wfo_help = "Perform a backtest with walk-forward optimization using the strategies defined in backtest_test.py\n"
	backtest_wfo_help += "The WFO years parameter specifies the size of the window into the recent past to perform parameter optimization with"
	backtest_wfo_help += "Requires the use of --start and --end"
	group.add_argument("--backtest-wfo", metavar=("SYMBOL", "WFO_YEARS"), nargs=2, help=backtest_wfo_help)

	args = parser.parse_args()
	if args.generate_all:
		generate_all_contracts()
	elif args.generate is not None:
		symbol: str = args.generate
		generate_contract(symbol)
	elif args.chart is not None:
		symbols: list[str] = args.chart
		start: pd.Timestamp | None = args.start
		end: pd.Timestamp | None = args.end
		render_chart(symbols, start, end)
	elif args.compare is not None:
		assert args.start is not None and args.end is not None
		symbols: list[str] = args.compare
		start: pd.Timestamp | None = args.start
		end: pd.Timestamp | None = args.end
		render_comparison_chart(symbols, start, end)
	elif args.chart_ratio is not None:
		assert args.start is not None and args.end is not None
		symbols: list[str] = args.chart_ratio
		base_symbol = symbols[0]
		dividend_symbol = symbols[1]
		divisor_symbol = symbols[2]
		start: pd.Timestamp | None = args.start
		end: pd.Timestamp | None = args.end
		render_ratio_chart(base_symbol, dividend_symbol, divisor_symbol, start, end)
	elif args.heatmap is not None:
		assert args.start is not None and args.end is not None
		symbol, x_axis, y_axis, quantiles_string = args.heatmap
		start: pd.Timestamp = args.start
		end: pd.Timestamp = args.end
		quantiles = int(quantiles_string)
		render_heatmap(symbol, start, end, x_axis, y_axis, quantiles)
	elif args.heatmap_all is not None:
		assert args.start is not None and args.end is not None
		symbol = args.heatmap_all
		start: pd.Timestamp = args.start
		end: pd.Timestamp = args.end
		statistics_only: bool = args.statistics
		render_heatmap_all(symbol, start, end, statistics_only)
	elif args.seasonality is not None:
		assert args.start is not None and args.split is not None and args.end is not None
		symbols: list[str] = args.seasonality
		start: pd.Timestamp = args.start
		split: pd.Timestamp = args.split
		end: pd.Timestamp = args.end
		analyze_seasonality(symbols, start, split, end)
	elif args.seasonality_chart is not None:
		assert args.start is not None and args.end is not None
		symbol, mode_string = cast(tuple[str, str], args.seasonality_chart)
		mode_dict = {
			"day": SeasonalityChartMode.DAY_OF_WEEK,
			"month": SeasonalityChartMode.MONTH,
			"turn": SeasonalityChartMode.TURN_OF_MONTH,
			"quarter": SeasonalityChartMode.QUARTER,
		}
		mode = mode_dict[mode_string]
		start: pd.Timestamp = args.start
		end: pd.Timestamp = args.end
		render_seasonality_chart(symbol, mode, start, end)
	elif args.momentum is not None:
		symbols = args.momentum
		analyze_momentum(symbols)
	elif args.backtest:
		assert args.start is not None and args.end is not None
		start: pd.Timestamp = args.start
		end: pd.Timestamp = args.end
		perform_backtest(start, end)
	elif args.backtest_wfo is not None:
		assert args.start is not None and args.end is not None
		symbol, wfo_years_string = args.backtest_wfo
		wfo_years = int(wfo_years_string)
		start: pd.Timestamp = args.start
		end: pd.Timestamp = args.end
		perform_wfo_backtest(symbol, start, end, wfo_years)
	else:
		parser.print_help()

if __name__ == "__main__":
	main()