"""Stage 1: build the estimation sample, fit the BART hurdle, run the HMC sampler.

Run it as a script, or step through it cell by cell: the `# %%` markers are
read as cells by VS Code, PyCharm and jupytext.

Everything stage 2 needs is written to outputs/<TAG>/, so the two stages never
have to agree on anything beyond the contents of that directory.
"""

# %%
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config as cfg
from src import features as ft
from src import regions

# Run parameters. Edit these.

N_CHAINS        = 4
PARALLEL_CHAINS = 4
ITER_WARMUP     = 500
ITER_SAMPLING   = 400
THIN            = 1
MAX_TREEDEPTH   = 12
ADAPT_DELTA     = 0.95
SEED            = 42

# BART hyper-parameters: the defaults of Chipman, George and McCulloch (2010).
BART_NTREE     = 200
BART_K         = 2.0
BART_POWER     = 2.0
BART_BASE      = 0.95
BART_NDPOST    = 1000
BART_NSKIP     = 500
BART_SUBSAMPLE = None      # None uses the full hurdle training set

# Country subset: 0 for the full panel, or 50 / 80 / 110 / 140.
RUN_SIZE = 0

# Stop after the BART fit and the integrity checks, without sampling.
SKIP_SAMPLING = False

TAG = "main" if RUN_SIZE == 0 else f"subset{RUN_SIZE}"
RUN_DIR = cfg.run_dir(TAG)
print(f"run '{TAG}' -> {RUN_DIR}")

# %%
# Sample

df = pd.read_csv(cfg.DATA_PATH)
df = df[df["orig"] != df["dest"]]
df = df[~df["orig"].isin(cfg.EXCLUDED_COUNTRIES)
        & ~df["dest"].isin(cfg.EXCLUDED_COUNTRIES)]

subset = cfg.SUBSETS[RUN_SIZE]
if subset is not None:
    df = df[df["orig"].isin(subset) & df["dest"].isin(subset)]

df = df.sort_values(["orig", "dest", "year"]).reset_index(drop=True)
df["dyad"] = df["orig"] + "_" + df["dest"]

n_countries = df["orig"].nunique()
print(f"{n_countries} countries, {len(df):,} dyad-years, "
      f"waves {sorted(df['year'].unique())}")

# %%
# M49 sub-region of the origin: the partition along which rho, phi and kappa
# are pooled.
cluster, m49_to_stan, stan_to_m49, K_CLUSTERS = regions.assign_clusters(df["orig"])
df["cluster"] = cluster
CLUSTER_LABELS = regions.cluster_labels(stan_to_m49, K_CLUSTERS)
print(f"{K_CLUSTERS} sub-regions represented")

df = ft.add_log_covariates(df, cfg.GRAVITY_VARS_RAW)
df = ft.add_flow_momentum(df)
print(f"momentum non-zero on {(df['flow_momentum'] != 0).mean() * 100:.1f}% of rows")

# %%
# Temporal split.
#
# The hurdle stops at 2005 so that 2010, where the per-cluster thresholds are
# chosen, is genuinely out of sample for it. The volume component runs to 2010
# because the zero-truncated likelihood has far fewer rows to work with; the
# consequence is that 2010 is not out of sample for the volume model, which is
# why the 'roc' calibration mode, which never looks at a predicted flow, exists
# as a robustness check.

df_train = df[df["year"] <= cfg.VOLUME_TRAIN_END].copy()
df_test = pd.concat(
    [df[df["year"] == cfg.CALIBRATION_YEAR], df[df["year"] == cfg.EVALUATION_YEAR]],
    ignore_index=True,
)
df_test["is_evaluation"] = (df_test["year"] == cfg.EVALUATION_YEAR).astype(int)

HURDLE_REQUIRED = ["is_migration", "dyad", "cluster", "is_mig_lag"]
hurdle_years = df_train["year"] <= cfg.HURDLE_TRAIN_END
if cfg.DROP_FIRST_WAVE_FROM_HURDLE:
    hurdle_years &= df_train["year"] > cfg.FIRST_YEAR
df_hurdle = (df_train[hurdle_years]
             .dropna(subset=HURDLE_REQUIRED)
             .reset_index(drop=True))

if not cfg.DROP_FIRST_WAVE_FROM_HURDLE:
    # Worth knowing about: in the first wave the lag is filled with zero for
    # every dyad, so is_mig_lag, log_stock_lag and flow_momentum carry no
    # information there and the corridor-open indicator is wrong for the half
    # of them that were in fact open.
    first = (df_hurdle["year"] == cfg.FIRST_YEAR)
    if first.any():
        print(f"note: {first.sum():,} of {len(df_hurdle):,} hurdle rows are the "
              f"{cfg.FIRST_YEAR} wave, where the lagged covariates are "
              f"degenerate (see cfg.DROP_FIRST_WAVE_FROM_HURDLE)")

# %%
# Volume sample: positive flows only, minus the re-openings and the first wave.
has_history, continuing, virgin, reopening = ft.markov_states(df_train)
df_train = df_train.assign(has_history=has_history)

VOLUME_REQUIRED = cfg.X_VOL_COLS + ["flow", "is_mig_lag", "has_history",
                                    "dyad", "cluster"]
positive = df_train["flow"] > 0
censored = df_train["year"] == cfg.FIRST_YEAR

n_reopen = int((reopening & positive & ~censored).sum())
df_volume = (df_train[positive & (continuing | virgin) & ~censored]
             .dropna(subset=VOLUME_REQUIRED)
             .reset_index(drop=True))
df_volume["is_emergent"] = (1 - df_volume["is_mig_lag"]).astype(int)
df_volume["log_flow_lag_clean"] = df_volume["log_flow_lag"].fillna(0.0)

print(f"hurdle {len(df_hurdle):,} rows, waves {sorted(df_hurdle['year'].unique())}")
print(f"volume {len(df_volume):,} rows ({n_reopen:,} re-openings excluded, "
      f"{cfg.FIRST_YEAR} excluded for lack of t-1)")

# %%
# Network features. Built after the split so that the degrees are counted on
# the hurdle training window only.
df_hurdle, df_test, df_volume = ft.add_degree_features(df_hurdle, df_test, df_volume)

isolated = (df_test["is_mig_lag"] == 0) & (df_test["log_stock_lag"] == 0)
print(f"dyads with neither a lagged flow nor a stock: {isolated.sum():,}, "
      f"of which {(df_test.loc[isolated, 'A2_log'] > 0).mean() * 100:.1f}% "
      f"have a two-step route")

# %%
# Test frame: drop the rows no component can score, then freeze it. Every index
# downstream refers to this ordering, so nothing may reorder or filter df_test
# past this point.

df_test = (df_test
           .replace([np.inf, -np.inf], np.nan)
           .dropna(subset=["log_gdpcap_d_lag"] + cfg.X_VOL_COLS + cfg.BART_VARS)
           .reset_index(drop=True))
df_test["log_flow_lag_clean"] = (
    df_test["log_flow_lag"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
)
df_test["cluster"] = (
    df_test["orig"].map(lambda x: regions.ISO3_TO_M49.get(str(x).upper(),
                                                          regions.UNCLASSIFIED))
    .map(m49_to_stan).fillna(K_CLUSTERS).astype(int)
)

n_cal = int((df_test["is_evaluation"] == 0).sum())
n_ev = int((df_test["is_evaluation"] == 1).sum())
if n_cal != n_ev:
    print(f"warning: calibration ({n_cal:,}) and evaluation ({n_ev:,}) years have "
          f"different sizes after the dropna; the two are never compared row-wise, "
          f"so this is safe, but it is worth a look")

df_hurdle = (df_hurdle.replace([np.inf, -np.inf], np.nan)
             .dropna(subset=HURDLE_REQUIRED).reset_index(drop=True))
df_volume = (df_volume.replace([np.inf, -np.inf], np.nan)
             .dropna(subset=VOLUME_REQUIRED).reset_index(drop=True))
print(f"test frame frozen at {len(df_test):,} rows "
      f"({n_cal:,} in {cfg.CALIBRATION_YEAR}, {n_ev:,} in {cfg.EVALUATION_YEAR})")

# %%
# Indices and design matrices

countries = sorted(set(df_train["orig"]) | set(df_train["dest"]))
country_to_id = {c: i + 1 for i, c in enumerate(countries)}
N_COUNTRIES = len(countries)

dyads_v = sorted(df_volume["dyad"].unique())
dyad_to_v = {d: i + 1 for i, d in enumerate(dyads_v)}
df_volume["dyad_id"] = df_volume["dyad"].map(dyad_to_v)
D_V = len(dyads_v)

# One cluster per dyad, taken from the origin, in dyad-id order.
cluster_v = (df_volume.groupby("dyad")["cluster"].first()
             .reindex(dyads_v).values.astype(int))

df_volume["orig_id"] = df_volume["orig"].map(country_to_id).astype(int)
df_volume["dest_id"] = df_volume["dest"].map(country_to_id).astype(int)

# Unknown test dyads get id 0, which the Stan model reads as "no dyadic random
# effect available, fall back on the cluster". A country absent from training
# cannot happen given the subsetting, but the fillna keeps the index inside the
# bounds Stan declares rather than crashing mid-run.
df_test["dyad_id"] = df_test["dyad"].map(dyad_to_v).fillna(0).astype(int)
df_test["orig_id"] = df_test["orig"].map(country_to_id).fillna(1).astype(int)
df_test["dest_id"] = df_test["dest"].map(country_to_id).fillna(1).astype(int)

X_vol, vol_stats = ft.standardize(
    df_volume[cfg.X_VOL_COLS].values, cfg.X_VOL_COLS, cfg.BINARY_COLS_VOL)
X_vol_test, _ = ft.standardize(
    df_test[cfg.X_VOL_COLS].values, cfg.X_VOL_COLS, cfg.BINARY_COLS_VOL,
    stats=vol_stats)

Z = ft.build_country_covariate(
    df_train[df_train["year"] == cfg.VOLUME_TRAIN_END],
    country_to_id, regions.ISO3_TO_M49)
print(f"Z: {N_COUNTRIES} countries, range [{Z.min():.2f}, {Z.max():.2f}]")

# Per-dyad average of the lagged level, standardised. Enters the model as the
# scale on which the re-opening intercept is expressed.
scale_v = df_volume.groupby("dyad")["log_flow_lag_clean"].mean().reindex(dyads_v).values
scale_v = (scale_v - scale_v.mean()) / max(scale_v.std(), 1e-8)

# %%
stan_data = {
    "N_pays": N_COUNTRIES,
    "K_Z": Z.shape[1],
    "Z_em": Z.tolist(),
    "Z_at": Z.tolist(),
    "K_clusters": K_CLUSTERS,

    "N_v": len(df_volume),
    "D_v": D_V,
    "K_v": len(cfg.X_VOL_COLS),
    "dyad_id_v": df_volume["dyad_id"].astype(int).tolist(),
    "orig_id_v": df_volume["orig_id"].astype(int).tolist(),
    "dest_id_v": df_volume["dest_id"].astype(int).tolist(),
    "flow": df_volume["flow"].astype(int).tolist(),
    "log_flow_lag": df_volume["log_flow_lag_clean"].astype(float).tolist(),
    "momentum_v": df_volume["flow_momentum"].astype(float).tolist(),
    "is_emergent_v": df_volume["is_emergent"].astype(int).tolist(),
    "X_v": X_vol.tolist(),
    "log_scale_v": scale_v.tolist(),
    "cluster_v": cluster_v.tolist(),

    "N_test": len(df_test),
    "dyad_id_test_v": df_test["dyad_id"].astype(int).tolist(),
    "orig_id_test_v": df_test["orig_id"].astype(int).tolist(),
    "dest_id_test_v": df_test["dest_id"].astype(int).tolist(),
    "X_v_test": X_vol_test.tolist(),
    "log_flow_lag_test": df_test["log_flow_lag_clean"].astype(float).tolist(),
    "momentum_test": df_test["flow_momentum"].astype(float).tolist(),
    "is_mig_lag_test": df_test["is_mig_lag"].fillna(0.0).astype(float).tolist(),
    "cluster_test": df_test["cluster"].astype(int).tolist(),

    "do_loo": 0,
}

assert stan_data["K_v"] == len(cfg.X_VOL_COLS)
assert len(stan_data["X_v_test"]) == stan_data["N_test"] == len(df_test)
assert len(stan_data["log_scale_v"]) == D_V
assert min(stan_data["dyad_id_test_v"]) >= 0
assert max(stan_data["cluster_test"]) <= K_CLUSTERS
ft.check_finite(stan_data)
print(f"stan_data ready: N_v = {stan_data['N_v']:,}, D_v = {D_V:,}, "
      f"N_test = {stan_data['N_test']:,}, all finite")

unknown = int((df_test["dyad_id"] == 0).sum())
print(f"test dyads absent from the volume training set: {unknown:,} "
      f"({unknown / len(df_test) * 100:.1f}%), routed through kappa")

# %%
# Hurdle: BART probit, fitted through R.
#
# The two components are separable, so this does not have to wait for the
# sampler. dbarts is called through rpy2 because no Python implementation of
# BART both handles a probit link and returns the posterior draws, and the
# draws are what the predictive intervals in stage 2 need.

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from sklearn.metrics import roc_auc_score

missing_train = [c for c in cfg.BART_VARS if c not in df_hurdle.columns]
missing_test = [c for c in cfg.BART_VARS if c not in df_test.columns]
assert not missing_train, f"absent from df_hurdle: {missing_train}"
assert not missing_test, f"absent from df_test: {missing_test}"

X_bart = df_hurdle[cfg.BART_VARS].fillna(0).values
y_bart = df_hurdle["is_migration"].values.astype(float)
X_bart_test = df_test[cfg.BART_VARS].fillna(0).values
assert X_bart_test.shape[0] == stan_data["N_test"]

y_reference = y_bart
if BART_SUBSAMPLE and BART_SUBSAMPLE < len(X_bart):
    take = np.random.default_rng(SEED).choice(len(X_bart), BART_SUBSAMPLE,
                                              replace=False)
    X_bart, y_bart = X_bart[take], y_bart[take]
    y_reference = y_bart

print(f"BART: {len(cfg.BART_VARS)} covariates, {X_bart.shape[0]:,} training rows")

ro.r("suppressMessages(library(dbarts))")
with localconverter(ro.default_converter + numpy2ri.converter):
    ro.globalenv["x_train"] = X_bart
    ro.globalenv["y_train"] = y_bart
    ro.globalenv["x_test"] = X_bart_test
ro.globalenv["var_names"] = ro.StrVector(cfg.BART_VARS)
# dbarts switches to a probit fit on a 0/1 response and to a Gaussian
# regression otherwise; the difference is silent, so check.
ro.r("stopifnot(identical(sort(unique(y_train)), c(0, 1)))")

for name, value in [("n_tree", BART_NTREE), ("k", float(BART_K)),
                    ("power", float(BART_POWER)), ("base", float(BART_BASE)),
                    ("n_post", BART_NDPOST), ("n_skip", BART_NSKIP),
                    ("seed", SEED)]:
    ro.globalenv[name] = value

t0 = time.time()
ro.r("""
    colnames(x_train) <- var_names
    colnames(x_test)  <- var_names
    set.seed(seed)
    fit <- bart(x.train = x_train, y.train = y_train, x.test = x_test,
                ntree = n_tree, k = k, power = power, base = base,
                ndpost = n_post, nskip = n_skip, verbose = FALSE)
    p_test  <- pnorm(fit$yhat.test)
    p_train <- pnorm(fit$yhat.train)
    inclusion <- fit$varcount
""")

p_bart = np.asarray(ro.r("p_test"), dtype=np.float32)
p_bart_train = np.median(np.asarray(ro.r("p_train")), axis=0)
bart_inclusion = np.asarray(ro.r("inclusion"), dtype=np.float32)
ro.r("rm(fit); invisible(gc())")

assert p_bart.shape[1] == len(df_test), "BART draws and df_test are misaligned"
df_test["p_hurdle"] = np.median(p_bart, axis=0)

is_cal = (df_test["is_evaluation"] == 0).values
is_ev = (df_test["is_evaluation"] == 1).values
auc = {
    "train": roc_auc_score(y_reference, p_bart_train),
    str(cfg.CALIBRATION_YEAR): roc_auc_score(
        (df_test.loc[is_cal, "flow"] > 0).astype(int), df_test.loc[is_cal, "p_hurdle"]),
    str(cfg.EVALUATION_YEAR): roc_auc_score(
        (df_test.loc[is_ev, "flow"] > 0).astype(int), df_test.loc[is_ev, "p_hurdle"]),
}
print(f"BART fitted in {time.time() - t0:.0f}s, draws {p_bart.shape} "
      f"({p_bart.nbytes / 1e6:.0f} MB)")
print("AUC  " + "  ".join(f"{k} {v:.4f}" for k, v in auc.items()))

# %%
# Persist everything stage 2 reads.

meta = {
    "tag": TAG,
    "n_countries": int(n_countries),
    "n_chains": N_CHAINS,
    "iter_warmup": ITER_WARMUP,
    "iter_sampling": ITER_SAMPLING,
    "seed": SEED,
    "K_clusters": K_CLUSTERS,
    "cluster_labels": CLUSTER_LABELS,
    "stan_to_m49": {str(k): v for k, v in stan_to_m49.items()},
    "x_vol_cols": cfg.X_VOL_COLS,
    "bart_vars": cfg.BART_VARS,
    "bart_auc": auc,
    "n_volume": int(len(df_volume)),
    "n_hurdle": int(len(df_hurdle)),
    "n_test": int(len(df_test)),
    "n_dyads_volume": int(D_V),
    "n_reopenings_excluded": n_reopen,
    "calibration_year": cfg.CALIBRATION_YEAR,
    "evaluation_year": cfg.EVALUATION_YEAR,
    "max_treedepth": MAX_TREEDEPTH,
    "test_signature": ft.frame_signature(df_test, cfg.X_VOL_COLS),
}

TEST_COLUMNS = [
    "orig", "dest", "year", "flow", "is_evaluation", "is_mig_lag",
    "cluster", "dyad", "dyad_id", "p_hurdle", "log_flow_lag_clean",
    "flow_momentum", "log_stock_lag",
]
df_test[TEST_COLUMNS].to_csv(RUN_DIR / "test_frame.csv.gz", index=False)
np.savez_compressed(RUN_DIR / "bart.npz",
                    p_test=p_bart, inclusion=bart_inclusion,
                    var_names=np.array(cfg.BART_VARS))
(RUN_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
print(f"artifacts written to {RUN_DIR}")

# %%
# Volume: HMC on the zero-truncated negative binomial ARX.

if SKIP_SAMPLING:
    print("SKIP_SAMPLING is set: stopping before the HMC run")
    sys.exit(0)

from cmdstanpy import CmdStanModel

model = CmdStanModel(stan_file=str(cfg.STAN_FILE))

t0 = time.time()
fit = model.sample(
    data=stan_data,
    chains=N_CHAINS,
    parallel_chains=PARALLEL_CHAINS,
    iter_warmup=ITER_WARMUP,
    iter_sampling=ITER_SAMPLING,
    thin=THIN,
    seed=SEED,
    inits=0.1,
    adapt_delta=ADAPT_DELTA,
    max_treedepth=MAX_TREEDEPTH,
    save_warmup=False,
    show_progress=True,
    sig_figs=6,
    output_dir=str(RUN_DIR / "stan_raw"),
)
print(f"sampled in {(time.time() - t0) / 60:.1f} min")

# Rename to a stable pattern so that stage 2 does not have to guess.
stan_dir = RUN_DIR / "stan"
stan_dir.mkdir(exist_ok=True)
csv_files = []
for i, produced in enumerate(fit.runset.csv_files):
    target = stan_dir / f"chain{i + 1}.csv"
    shutil.move(produced, target)
    csv_files.append(target.name)
shutil.rmtree(RUN_DIR / "stan_raw", ignore_errors=True)

meta["stan_csv"] = csv_files
meta["stan_file"] = cfg.STAN_FILE.name
(RUN_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

print(f"\n{len(csv_files)} chains in {stan_dir}")
print(f"next: set TAG = '{TAG}' in 02_analyse_draws.py and run it")
