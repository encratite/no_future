from typing import Final

from filter import ContractFilter

def get_contract_filter(barchart_symbol: str) -> ContractFilter | None:
	if barchart_symbol in CONTRACT_CONFIGURATION:
		return CONTRACT_CONFIGURATION[barchart_symbol]
	else:
		return None

def get_barchart_symbol(exchange_symbol: str) -> str:
	for barchart_symbol, contract_filter in CONTRACT_CONFIGURATION.items():
		if contract_filter.exchange_symbol == exchange_symbol:
			return barchart_symbol
		if barchart_symbol == exchange_symbol and contract_filter.exchange_symbol is None:
			return exchange_symbol
	raise Exception(f"Unknown exchange symbol: {exchange_symbol}")

CONTRACT_CONFIGURATION: Final[dict[str, ContractFilter]] = {
	"A6": ContractFilter("6A", legacy_cutoff="A6H01", first_filter_contract="A6J17", include_months=["H", "M", "U", "Z"], cutoff_date="2001-03-27"),
	"B6": ContractFilter("6B", legacy_cutoff="B6M03", first_filter_contract="B6J17", include_months=["H", "M", "U", "Z"]),
	"BA": ContractFilter("MBT"),
	"BT": ContractFilter("BTC"),
	"CL": ContractFilter(legacy_cutoff="CLK03", cutoff_date="2000-09-01"),
	"CN": ContractFilter("MNG"),
	"CT": ContractFilter(exclude_months=["V"]),
	"CY": ContractFilter("MCL"),
	"DL": ContractFilter("DA", cutoff_date="2006-10-06"),
	"DV": ContractFilter("V2TX"),
	"DY": ContractFilter("FDAX", copy="FDXM", legacy_cutoff="DYM02"),
	"D6": ContractFilter("6C", legacy_cutoff="D6H01", first_filter_contract="D6J17", include_months=["H", "M", "U", "Z"]),
	"E6": ContractFilter("6E", legacy_cutoff="E6H02", first_filter_contract="E6J17", include_months=["H", "M", "U", "Z"], cutoff_date="2001-11-24"),
	"ER": ContractFilter("ETH"),
	"ES": ContractFilter(legacy_cutoff="ESU02"),
	"ET": ContractFilter("MES"),
	"GC": ContractFilter(copy="MGC", legacy_cutoff="GCG06", exclude_months=["F", "H", "K", "N", "U", "V", "X"]),
	"GR": ContractFilter("MGC", exclude_months=["V"]),
	"HE": ContractFilter(legacy_cutoff="HEJ02", cutoff_date="2002-03-02"),
	"HG": ContractFilter(legacy_cutoff="HGK03", include_months=["H", "K", "N", "U", "Z"]),
	"HO": ContractFilter(legacy_cutoff="HOG01", cutoff_date="2000-05-04"),
	"J6": ContractFilter("6J", legacy_cutoff="J6Z01", first_filter_contract="J6J17", include_months=["H", "M", "U", "Z"]),
	"JM": ContractFilter("FXDS"),
	"LE": ContractFilter(legacy_cutoff="LEG02", first_filter_contract="LEK03", last_filter_contract="LEK05", exclude_months=["F", "H", "K", "N", "U", "X"], cutoff_date="2003-10-02"),
	"MF": ContractFilter("M6E"),
	"NG": ContractFilter(legacy_cutoff="NGF04"),
	"NM": ContractFilter("MNQ"),
	"NQ": ContractFilter(legacy_cutoff="NQM01"),
	"PL": ContractFilter(legacy_cutoff="PLN01", first_filter_contract="PLG10", exclude_months=["F", "J", "N", "V"], cutoff_date="2002-08-08"),
	"QL": ContractFilter("MHG"),
	"QR": ContractFilter(legacy_cutoff="QRH03"),
	"RB": ContractFilter(legacy_cutoff="RBJ06"),
	"RX": ContractFilter(legacy_cutoff="RXM19"),
	"S6": ContractFilter("6S", legacy_cutoff="S6H02", first_filter_contract="S6J17", include_months=["H", "M", "U", "Z"]),
	"SB": ContractFilter(legacy_cutoff="SBH03"),
	"SI": ContractFilter(legacy_cutoff="SIH03", include_months=["F", "G", "J", "M", "Q", "V", "X"], f_records_limit=2),
	"SO": ContractFilter("SIL", exclude_months=["F", "G", "J", "M", "Q", "V", "X"]),
	"TA": ContractFilter("MET"),
	"TM": ContractFilter("FDXM"),
	"VI": ContractFilter("VX", enable_fy_records=False),
	"VJ": ContractFilter("VXM"),
	"WN": ContractFilter("MSF"),
	"XK": ContractFilter(exclude_months=["Q", "U"]),
	"XN": ContractFilter("XC", exclude_months=["U"]),
	"ZB": ContractFilter(legacy_cutoff="ZBM02", cutoff_date="2004-11-12"),
	"ZC": ContractFilter(legacy_cutoff="ZCK02"),
	"ZF": ContractFilter(legacy_cutoff="ZFM02"),
	"ZL": ContractFilter(legacy_cutoff="ZLQ02"),
	"ZM": ContractFilter(legacy_cutoff="ZMQ02", cutoff_date="2002-11-20"),
	"ZN": ContractFilter(legacy_cutoff="ZNU01", cutoff_date="2004-11-11", enable_fy_records=False),
	"ZQ": ContractFilter(legacy_cutoff="ZQF01"),
	"ZS": ContractFilter(legacy_cutoff="ZSK02", cutoff_date="2001-08-04"),
	"ZT": ContractFilter(legacy_cutoff="ZTH02", enable_fy_records=False),
	"ZW": ContractFilter(legacy_cutoff="ZWK02")
}