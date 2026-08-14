"""UN M49 sub-region lookup and the display labels used in figures.

The M49 sub-region of the origin country is the partition along which the
model pools the AR(1) inertia, the negative binomial dispersion and the
re-opening intercept. Codes follow the UN Statistics Division geoscheme.
"""


ISO3_TO_M49 = {
    # 011 Northern Europe
    "DNK": 11, "EST": 11, "FIN": 11, "ISL": 11, "IRL": 11, "LVA": 11,
    "LTU": 11, "NOR": 11, "SWE": 11, "GBR": 11,
    # 012 Southern Europe
    "ALB": 12, "AND": 12, "BIH": 12, "HRV": 12, "GRC": 12, "ITA": 12,
    "MLT": 12, "MNE": 12, "MKD": 12, "PRT": 12, "SRB": 12, "SVN": 12, "ESP": 12,
    # 013 Western Europe
    "AUT": 13, "BEL": 13, "FRA": 13, "DEU": 13, "LIE": 13, "LUX": 13,
    "MCO": 13, "NLD": 13, "CHE": 13,
    # 014 Eastern Europe
    "BLR": 14, "BGR": 14, "CZE": 14, "HUN": 14, "POL": 14, "MDA": 14,
    "ROU": 14, "RUS": 14, "SVK": 14, "UKR": 14,
    # 015 Northern Africa
    "DZA": 15, "EGY": 15, "LBY": 15, "MAR": 15, "SDN": 15, "TUN": 15, "ESH": 15,
    # 016 Western Africa
    "BEN": 16, "BFA": 16, "CPV": 16, "CIV": 16, "GMB": 16, "GHA": 16,
    "GIN": 16, "GNB": 16, "LBR": 16, "MLI": 16, "MRT": 16, "NER": 16,
    "NGA": 16, "SEN": 16, "SLE": 16, "TGO": 16,
    # 017 Eastern Africa
    "BDI": 17, "COM": 17, "DJI": 17, "ERI": 17, "ETH": 17, "KEN": 17,
    "MDG": 17, "MWI": 17, "MUS": 17, "MOZ": 17, "REU": 17, "RWA": 17,
    "SYC": 17, "SOM": 17, "SSD": 17, "TZA": 17, "UGA": 17, "ZMB": 17, "ZWE": 17,
    # 018 Middle Africa
    "AGO": 18, "CMR": 18, "CAF": 18, "TCD": 18, "COD": 18, "COG": 18,
    "GNQ": 18, "GAB": 18, "STP": 18,
    # 019 Southern Africa
    "BWA": 19, "LSO": 19, "NAM": 19, "ZAF": 19, "SWZ": 19,
    # 021 Northern America
    "CAN": 21, "MEX": 21, "USA": 21,
    # 022 Central America
    "BLZ": 22, "CRI": 22, "SLV": 22, "GTM": 22, "HND": 22, "NIC": 22, "PAN": 22,
    # 023 Caribbean
    "ATG": 23, "BHS": 23, "BRB": 23, "CUB": 23, "DMA": 23, "DOM": 23,
    "GLP": 23, "GRD": 23, "HTI": 23, "JAM": 23, "KNA": 23, "LCA": 23,
    "MTQ": 23, "VCT": 23, "TTO": 23, "ABW": 23, "PRI": 23,
    # 024 South America
    "ARG": 24, "BOL": 24, "BRA": 24, "CHL": 24, "COL": 24, "ECU": 24,
    "GUF": 24, "GUY": 24, "PRY": 24, "PER": 24, "SUR": 24, "URY": 24, "VEN": 24,
    # 030 Eastern Asia
    "CHN": 30, "HKG": 30, "JPN": 30, "KOR": 30, "MAC": 30, "MNG": 30, "PRK": 30,
    # 034 Southern Asia
    "AFG": 34, "BGD": 34, "BTN": 34, "IND": 34, "IRN": 34, "MDV": 34,
    "NPL": 34, "PAK": 34, "LKA": 34,
    # 035 South-eastern Asia
    "BRN": 35, "KHM": 35, "IDN": 35, "LAO": 35, "MYS": 35, "MMR": 35,
    "PHL": 35, "SGP": 35, "THA": 35, "TLS": 35, "VNM": 35,
    # 143 Central Asia
    "KAZ": 143, "KGZ": 143, "TJK": 143, "TKM": 143, "UZB": 143,
    # 145 Western Asia
    "ARM": 145, "AZE": 145, "BHR": 145, "CYP": 145, "GEO": 145, "IRQ": 145,
    "ISR": 145, "JOR": 145, "KWT": 145, "LBN": 145, "OMN": 145, "QAT": 145,
    "SAU": 145, "PSE": 145, "SYR": 145, "TUR": 145, "ARE": 145, "YEM": 145,
    # 053 Oceania (Australia and New Zealand, Melanesia, Micronesia, Polynesia
    # are merged: the flows involving the small island states are too sparse to
    # identify four separate variance components)
    "AUS": 53, "FJI": 53, "NZL": 53, "PNG": 53, "SLB": 53, "VUT": 53,
    "WSM": 53, "TON": 53, "KIR": 53, "FSM": 53, "GUM": 53, "NCL": 53, "PYF": 53,
}

UNCLASSIFIED = 99

M49_LABELS = {
    11: "Northern Europe",
    12: "Southern Europe",
    13: "Western Europe",
    14: "Eastern Europe",
    15: "Northern Africa",
    16: "Western Africa",
    17: "Eastern Africa",
    18: "Middle Africa",
    19: "Southern Africa",
    21: "Northern America",
    22: "Central America",
    23: "Caribbean",
    24: "South America",
    30: "Eastern Asia",
    34: "Southern Asia",
    35: "South-eastern Asia",
    53: "Oceania",
    143: "Central Asia",
    145: "Western Asia",
    UNCLASSIFIED: "Unclassified",
}


def assign_clusters(iso3_series):
    """Map ISO3 codes to contiguous 1..K cluster ids that Stan can index.

    Returns the cluster id per row, the id -> M49 code map and K. Only the
    sub-regions actually present in the sample get an id, so K shrinks with
    the country subset.
    """
    m49 = iso3_series.map(lambda x: ISO3_TO_M49.get(str(x).upper(), UNCLASSIFIED))
    present = sorted(m49.unique())
    m49_to_stan = {code: i + 1 for i, code in enumerate(present)}
    stan_to_m49 = {v: k for k, v in m49_to_stan.items()}
    return m49.map(m49_to_stan), m49_to_stan, stan_to_m49, len(m49_to_stan)


def cluster_labels(stan_to_m49, n_clusters):
    return [
        M49_LABELS.get(stan_to_m49.get(k, UNCLASSIFIED), f"cluster {k}")
        for k in range(1, n_clusters + 1)
    ]


# Display names for covariates. Written as LaTeX maths where the paper uses
# maths, so that the figures and the equations agree glyph for glyph.

COVARIATE_LABELS = {
    "log_D_ij": r"$\log D_{ij}$",
    "log_D_ij_sq": r"$(\log D_{ij})^{2}$",
    "LB_ij": "Shared border",
    "OL_ij": "Common language",
    "COL_ij": "Colonial tie",
    "logD_times_LB": r"$\log D_{ij}\times$ border",
    "v2x_polyarchy_o_lag5": r"Polyarchy, origin $(t{-}1)$",
    "v2x_polyarchy_d_lag5": r"Polyarchy, destination $(t{-}1)$",
    "v2x_clphy_o_lag5": r"Physical integrity, origin $(t{-}1)$",
    "v2x_clphy_d_lag5": r"Physical integrity, destination $(t{-}1)$",
    "intensity_level_o_lag5": r"Conflict intensity, origin $(t{-}1)$",
    "intensity_level_d_lag5": r"Conflict intensity, destination $(t{-}1)$",
    "log_gdpcap_o_lag5": r"$\log$ GDP per capita, origin $(t{-}1)$",
    "log_gdpcap_d_lag5": r"$\log$ GDP per capita, destination $(t{-}1)$",
    "log_P_it": r"$\log P_{it}$",
    "log_P_jt": r"$\log P_{jt}$",
    "PSR_i": "Potential support ratio, origin",
    "PSR_j": "Potential support ratio, destination",
    "IMR_it": "Infant mortality, origin",
    "IMR_jt": "Infant mortality, destination",
    "urban_it": "Urban share, origin",
    "urban_jt": "Urban share, destination",
    "LL_i": "Landlocked, origin",
    "LL_j": "Landlocked, destination",
    "LA_i": "Land area, origin",
    "LA_j": "Land area, destination",
    "is_mig_lag": r"Corridor open at $t{-}1$",
    "log_stock_lag": r"$\log$ migrant stock $(t{-}1)$",
    "transitivity_proxy": "Degree product",
    "A2_log": r"$\log(1+A^{2}_{ij})$",
    "flow_momentum": r"Flow momentum $\Delta_{ij,t-1}$",
}


def pretty(name: str) -> str:
    return COVARIATE_LABELS.get(name, name.replace("_", " "))
