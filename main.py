import datetime as dt
from argparse import ArgumentParser
from typing import cast

import pandas as pd

from chart import *
from generate import (
	generate_all_contracts,
	generate_contract,
	generate_intraday_contract
)
from heatmap import render_heatmap, render_heatmap_all
from intraday import analyze_session_returns
from momentum import analyze_momentum
from seasonality import analyze_seasonality
from test.test_quantile import perform_backtest
from test.test_wfo import perform_wfo_backtest
from z_score import analyze_z_score_pattern
from features import analyze_ohlc_features, FilterMode
from clustering import analyze_clusters

def get_date_argument(date_string: str) -> pd.Timestamp:
	pd.to_datetime(date_string, format="%Y-%m-%d", errors="raise")
	return pd.Timestamp(date_string)

def main() -> None:
	parser = ArgumentParser(description="Futures backtesting and analysis")
	group = parser.add_mutually_exclusive_group()
	group.add_argument("--generate-all", action="store_true", help="Generate continuous contracts for all symbols")
	group.add_argument("--generate", metavar="SYMBOL", help="Generate a continuous contract for the specified symbol")

	generate_intraday_help = "Generate an intraday continuous contract for the specified symbol\n"
	generate_intraday_help += "The suffix specifies the time frame to generate the Feather file from (H1, M30, M15, etc.)"
	group.add_argument("--generate-intraday", metavar=("SYMBOL", "SUFFIX"), nargs=2, help=generate_intraday_help)

	chart_help = "Render a chart for the specified symbols\n"
	chart_help += "By default this will display all available data\n"
	chart_help += "Use --start and --end to limit the chart to a certain time range"
	group.add_argument("--chart", metavar="SYMBOLS", nargs="*", help=chart_help)
	parser.add_argument("--intraday", action="store_true", help="Enables the intraday mode of --chart")

	compare_help = "Compare the specified symbols, with all initial prices normalized to 1.0\n"
	compare_help += "Use --start and --end to limit the chart to a certain time range"
	group.add_argument("--compare", metavar="SYMBOLS", nargs="*", help=compare_help)

	chart_ratio_help = "Render a ratio chart using the specified symbols\n"
	chart_ratio_help += "Use --start and --end to limit the chart to a certain time range"
	group.add_argument("--chart-ratio", metavar=("BASE", "DIVISOR", "DIVIDEND"), nargs=3, help=chart_ratio_help)

	chart_volatility_help = "Render volatility chart for the specified symbol using a window size of n days\n"
	chart_volatility_help += "Requires the use of --start and --end"
	group.add_argument("--volatility", metavar=("SYMBOL", "WINDOW"), nargs=2, help=chart_volatility_help)

	heatmap_help = "Render a heatmap of one-day returns of the specified symbol based on the quantiles of the two specified features\n"
	heatmap_help += "Supported features: momentum2, momentum3, momentum10, regime, volume, interest, volatility\n"
	heatmap_help += "The number of quantiles determines the number of cells in the heatmap\n"
	heatmap_help += "Requires the use of --start and --end"
	group.add_argument("--heatmap", metavar=("SYMBOL", "FEATURE1", "FEATURE2", "QUANTILES"), nargs=4, help=heatmap_help)

	heatmap_all_help = "Render all heatmaps for the specified symbol\n"
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

	momentum_help = "Analyze momentum correlation of a symbol\n"
	momentum_help += "Optionally supports --start and --end to restrict the time range to analyze"
	group.add_argument("--momentum", metavar="SYMBOLS", nargs="*", help=momentum_help)

	backtest_help = "Perform a backtest using the strategies defined in backtest_test.py\n"
	backtest_help += "Requires the use of --start and --end"
	group.add_argument("--backtest", action="store_true", help=backtest_help)

	backtest_wfo_help = "Perform a backtest with walk-forward optimization using the strategies defined in backtest_test.py\n"
	backtest_wfo_help += "The WFO years parameter specifies the size of the window into the recent past to perform parameter optimization with\n"
	backtest_wfo_help += "Requires the use of --start and --end"
	group.add_argument("--backtest-wfo", metavar=("SYMBOL", "WFO_YEARS"), nargs=2, help=backtest_wfo_help)

	session_returns_help = "Analyze intraday returns during a particular session\n"
	session_returns_help += "Requires the use of --start and --end"
	group.add_argument("--session-returns", metavar=("SYMBOL", "START", "END"), nargs=3, help=session_returns_help)

	z_score_help = "Analyze the distribution momentum Z-scores of two assets\n"
	z_score_help += "Requires the use of --start and --end\n"
	z_score_help += "Use --detailed to increase the accuracy of the heatmap for assets that require it\n"
	group.add_argument("--z-score", metavar=("SYMBOL1", "SYMBOL2"), nargs=2, help=z_score_help)
	parser.add_argument("--detailed", action="store_true", help="Enables the detailed heatmap mode of --z-score")
	parser.add_argument("--delay", action="store_true", help="Delay the time series of symbol 2 by one day to account for exchanges that are located in different timezones")
	parser.add_argument("--boundary", metavar="BOUNDARY", type=float, help="Z-score boundary for the default tiling mode of --z-score")
	parser.add_argument("--minimum", metavar="MINIMUM", type=float, help="Minimum value for all Z-scores in --z-score")
	parser.add_argument("--maximum", metavar="MAXIMUM", type=float, help="Maximum value for all Z-scores in --z-score")

	features_help = "Perform a basic analysis of OHLC features for various assets\n"
	features_help += "Requires the use of --start, --split and --end"
	group.add_argument("--features", metavar="SYMBOLS", nargs="*", help=features_help)

	filter_help = "Add a filter to --features\n"
	filter_help += "Supported values: positive, negative"
	parser.add_argument("--filter", metavar="FILTER", help=filter_help)
	parser.add_argument("--pca", metavar="FEATURES", type=int, help="Perform dimension reduction using PCA")
	parser.add_argument("--select-k-best", metavar="FEATURES", type=int, help="Perform dimension reduction using mutual information regression")

	clustering_help = "Perform feature cluster analysis on all permutations of the specified symbols\n"
	clustering_help += "Requires the use of --start, --split, --end and --cluster-size"
	group.add_argument("--clustering", metavar="SYMBOLS", nargs="*", help=clustering_help)
	parser.add_argument("--cluster-size", metavar="FEATURES", type=int, help="The number of clusters to be calculated for --clustering")

	args = parser.parse_args()
	if args.generate_all:
		generate_all_contracts()
	elif args.generate is not None:
		symbol: str = args.generate
		generate_contract(symbol)
	elif args.generate_intraday is not None:
		symbol, suffix = args.generate_intraday
		generate_intraday_contract(symbol, suffix)
	elif args.chart is not None:
		symbols: list[str] = args.chart
		start: pd.Timestamp | None = args.start
		end: pd.Timestamp | None = args.end
		intraday = args.intraday
		render_chart(symbols, start, end, intraday)
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
	elif args.volatility is not None:
		assert args.start is not None and args.end is not None
		symbol, window_size_string = args.volatility
		window_size = int(window_size_string)
		start: pd.Timestamp | None = args.start
		end: pd.Timestamp | None = args.end
		render_volatility_chart(symbol, window_size, start, end)
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
		start: pd.Timestamp | None = args.start
		end: pd.Timestamp | None = args.end
		analyze_momentum(symbols, start, end)
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
	elif args.session_returns is not None:
		assert args.start is not None and args.end is not None
		symbol, session_start_string, session_end_string = args.session_returns
		session_start = get_time(session_start_string)
		session_end = get_time(session_end_string)
		start: pd.Timestamp = args.start
		end: pd.Timestamp = args.end
		analyze_session_returns(symbol, session_start, session_end, start, end)
	elif args.z_score:
		assert args.start is not None and args.end is not None
		symbol1, symbol2 = args.z_score
		start: pd.Timestamp = args.start
		end: pd.Timestamp = args.end
		detailed: bool = args.detailed
		delay: bool = args.delay
		boundary: float | None = args.boundary
		minimum: float | None = args.minimum
		maximum: float | None = args.maximum
		analyze_z_score_pattern(symbol1, symbol2, start, end, detailed, delay, boundary, minimum, maximum)
	elif args.features is not None:
		assert args.start is not None and args.split is not None and args.end is not None
		symbols: list[str] = args.features
		start: pd.Timestamp = args.start
		split: pd.Timestamp = args.split
		end: pd.Timestamp = args.end
		filter_modes = {
			"positive": FilterMode.POSITIVE,
			"negative": FilterMode.NEGATIVE,
			"high5": FilterMode.NEW_HIGH_5,
			"low5": FilterMode.NEW_LOW_5
		}
		filter_mode_string = args.filter
		if filter_mode_string is None:
			filter_mode = FilterMode.NONE
		else:
			filter_mode = filter_modes[args.filter]
		pca_features = args.pca
		select_k_best = args.select_k_best
		analyze_ohlc_features(symbols, start, split, end, filter_mode, pca_features, select_k_best)
	elif args.clustering:
		assert args.start is not None and args.split is not None and args.end is not None
		assert args.cluster_size
		symbols: list[str] = args.clustering
		clusters: int = args.cluster_size
		start: pd.Timestamp = args.start
		split: pd.Timestamp = args.split
		end: pd.Timestamp = args.end
		analyze_clusters(symbols, clusters, start, split, end)
	else:
		parser.print_help()

def get_time(time_string: str) -> dt.time:
	return dt.datetime.strptime(time_string, "%H:%M").time()

if __name__ == "__main__":
	main()