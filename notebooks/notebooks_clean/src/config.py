"""Paths, sample definition and covariate lists.

The sampler settings and the calibration knobs live at the top of the two
scripts, where they are meant to be edited. What is here is what defines the
data rather than the run.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Paths. Edit these.

DATA_PATH  = REPO_ROOT.parent.parent / "data" / "panel_june_filled.csv"
STAN_FILE  = REPO_ROOT.parent.parent / "STAN" / "HMC_BART_vectorized.stan"
FIGURE_DIR = REPO_ROOT / "figures"
OUTPUT_DIR = REPO_ROOT / "outputs"


def run_dir(tag):
    """One directory per run: Stan CSVs, BART draws, and the frozen test frame."""
    d = OUTPUT_DIR / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


# Countries with gaps in the covariates that no imputation can honestly fill.
EXCLUDED_COUNTRIES = {"SSD", "CUW", "GUM", "MYT", "VIR", "CLI"}

# The panel is five-yearly: 1990, 1995, ..., 2015.
FIRST_YEAR = 1990
HURDLE_TRAIN_END = 2005   # the hurdle stops here so that 2010 is a genuine OOS year
VOLUME_TRAIN_END = 2010
CALIBRATION_YEAR = 2010   # where the per-cluster thresholds are chosen
EVALUATION_YEAR = 2015    # where every number in the paper is computed

# The 1990 wave carries no t-1, so the lagged inertia covariates are
# mechanically zero there. The volume model always drops it, since no AR term is
# defined. Set this to True to drop it from the hurdle as well; the published
# results keep it, hence the default.
DROP_FIRST_WAVE_FROM_HURDLE = False

# Volume component (zero-truncated negative binomial, gravity part).
X_VOL_COLS = [
    "log_D_ij",
    "log_D_ij_sq",
    "LB_ij",
    "OL_ij",
    "COL_ij",
    "v2x_polyarchy_o_lag5",
    "intensity_level_o_lag5",
    "v2x_clphy_d_lag5",
    "intensity_level_d_lag5",
]

# Binary columns are left on their 0/1 scale rather than standardised.
BINARY_COLS_VOL = ["LB_ij", "OL_ij", "COL_ij"]

# Hurdle component (BART probit). Deliberately wider than the volume set: BART
# selects, so there is no cost to offering it the full menu.
BART_VARS = [
    "log_D_ij",
    "log_D_ij_sq",
    "OL_ij",
    "COL_ij",
    "LB_ij",
    "log_gdpcap_o_lag5",
    "log_gdpcap_d_lag5",
    "log_P_it",
    "log_P_jt",
    "PSR_i",
    "PSR_j",
    "IMR_it",
    "IMR_jt",
    "urban_it",
    "urban_jt",
    "LL_i",
    "LL_j",
    "LA_i",
    "LA_j",
    "v2x_polyarchy_o_lag5",
    "v2x_polyarchy_d_lag5",
    "is_mig_lag",
    "log_stock_lag",
    "transitivity_proxy",
    "A2_log",
    "flow_momentum",
]

# Raw columns that get a log transform before use.
GRAVITY_VARS_RAW = [
    "P_it",
    "P_jt",
    "PSR_i",
    "PSR_j",
    "IMR_it",
    "IMR_jt",
    "urban_it",
    "urban_jt",
    "LA_i",
    "LA_j",
]

# Country subsets, used to check that the results are not driven by sample size.
SUBSET_50 = {
    "USA", "CAN", "MEX", "BRA", "ARG", "COL", "CHL", "PER", "VEN",
    "FRA", "DEU", "GBR", "ITA", "ESP", "POL", "RUS", "UKR", "SWE", "NLD", "ROU",
    "CHN", "IND", "JPN", "KOR", "IDN", "PAK", "BGD", "PHL", "VNM", "TUR",
    "IRN", "SAU", "THA", "MYS", "KAZ",
    "NGA", "ETH", "EGY", "COD", "ZAF", "TZA", "KEN", "DZA", "MAR", "GHA",
    "CIV", "AGO", "SEN",
    "AUS", "NZL",
}

SUBSET_80 = SUBSET_50 | {
    "NOR", "FIN", "DNK", "CHE", "AUT", "BEL", "GRC", "CZE",
    "BOL", "ECU", "URY", "GTM", "CUB", "DOM",
    "IRQ", "ISR", "ARE", "UZB", "MMR", "LKA", "NPL", "AFG",
    "CMR", "MLI", "BFA", "MOZ", "ZMB", "RWA", "TUN", "SDN",
}

SUBSET_110 = SUBSET_80 | {
    "HUN", "PRT", "IRL", "BGR", "SRB", "HRV", "BLR", "SVK",
    "HND", "SLV", "NIC", "CRI", "PAN", "PRY",
    "JOR", "LBN", "KWT", "OMN", "YEM", "KHM", "SGP",
    "TCD", "NER", "GIN", "BDI", "SOM", "MWI", "COG", "GAB", "NAM",
}

SUBSET_140 = SUBSET_110 | {
    "LTU", "LVA", "EST", "SVN", "MKD", "BIH", "ALB", "MDA",
    "HTI", "JAM", "TTO", "BHS", "GUY", "SUR",
    "QAT", "BHR", "SYR", "TJK", "KGZ", "LAO", "MNG",
    "LBY", "MRT", "TGO", "BEN", "LBR", "SLE", "CAF",
    "PNG", "FJI",
}

SUBSETS = {50: SUBSET_50, 80: SUBSET_80, 110: SUBSET_110, 140: SUBSET_140, 0: None}
