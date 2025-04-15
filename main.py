from argparse import ArgumentParser
import pandas as pd

from generate_contracts import generate_all_contracts, generate_contract
from chart import render_chart
from seasonality import analyze_seasonality
from regression import perform_regression

def get_date_argument(date_string: str) -> pd.Timestamp:
	pd.to_datetime(date_string, format="%Y-%m-%d", errors="raise")
	return pd.Timestamp(date_string)

parser = ArgumentParser(description="Futures backtesting environment")
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

regression_help = "Perform regression test using the specified symbols\n"
regression_help += "Requires the use of --start, --split and --end"
group.add_argument("--regression", metavar="SYMBOLS", nargs="*", help=regression_help)

parser.add_argument("--start", metavar="DATE", type=get_date_argument, help="Restrict data to be read to after this date")
parser.add_argument("--split", metavar="DATE", type=get_date_argument, help="Date at which to split in-sample and out-of-sample data")
parser.add_argument("--end", metavar="DATE", type=get_date_argument, help="Read no data after this date")

args, _ = parser.parse_known_args()
if args.regression is not None:
	reduction_group = parser.add_mutually_exclusive_group()
	reduction_group.add_argument("--pca", metavar="FEATURES", type=int, help="Apply PCA dimensionality reduction")
	reduction_group.add_argument("--select-k-best", metavar="FEATURES", type=int, help="Apply dimensionality reduction using mutual information regression")

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
elif args.regression is not None:
	assert args.start is not None and args.split is not None and args.end is not None
	symbols: list[str] = args.regression
	start: pd.Timestamp = args.start
	split: pd.Timestamp = args.split
	end: pd.Timestamp = args.end
	pca: int | None = args.pca
	select_k_best: int | None = args.select_k_best
	assert not pca or not select_k_best
	perform_regression(symbols, start, split, end, pca, select_k_best)
else:
	parser.print_help()