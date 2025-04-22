from argparse import ArgumentParser
from typing import cast

import pandas as pd

from backtest_test import perform_backtest
from chart import render_chart
from chart_seasonality import render_seasonality_chart, SeasonalityChartMode
from generate import generate_all_contracts, generate_contract
from momentum import analyze_momentum
from seasonality import analyze_seasonality

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

	seasonality_help = "Analyze seasonality of the specified symbols\n"
	seasonality_help += "Requires the use of --start, --split and --end"
	group.add_argument("--seasonality", metavar="SYMBOLS", nargs="*", help=seasonality_help)

	seasonality_chart_help = "Render a chart with cumulative seasonality time series for the specified symbol\n"
	seasonality_chart_help += "Supported modes: day, month, quarter\n"
	seasonality_chart_help += "Also requires the use of --start and --end"
	group.add_argument("--seasonality-chart", metavar=("SYMBOL", "MODE"), nargs="*", help=seasonality_chart_help)

	parser.add_argument("--start", metavar="DATE", type=get_date_argument, help="Restrict data to be read to after this date")
	parser.add_argument("--split", metavar="DATE", type=get_date_argument, help="Date at which to split in-sample and out-of-sample data")
	parser.add_argument("--end", metavar="DATE", type=get_date_argument, help="Read no data after this date")

	parser.add_argument("--momentum", metavar="SYMBOLS", nargs="*", help="Analyze momentum correlation of a symbol")

	backtest_help = "Perform a backtest using the strategies defined in backtest_test.py\n"
	backtest_help += "Requires the use of --start and --end"
	group.add_argument("--backtest", action="store_true", help=backtest_help)

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
	else:
		parser.print_help()

if __name__ == "__main__":
	main()