"""Stage 2: read the draws, calibrate, score, and draw every figure in the paper.

Reads outputs/<TAG>/ and writes figures/ and outputs/<TAG>/tables/. Nothing
here refits anything, so it is cheap to re-run while iterating on a figure.

Figures carry no title and no caption: the caption belongs in the .tex, and a
title inside the figure would only duplicate it in the wrong typeface.
"""

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config as cfg
from src import diagnostics as diag
from src import features as ft
from src import figstyle as fs
from src import regions

fs.use_style()
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# Run parameters. Edit these.

TAG = "main"               # which outputs/<TAG>/ to read

# 'cost' minimises MAE + LAMBDA * 100 * MAPE on the calibration year.
# 'roc' maximises a weighted Youden index and never touches a predicted volume,
# which makes it immune to the fact that 2010 is in the volume training window.
CALIBRATION_MODE = "cost"
LAMBDA           = 3.0
W_FN, W_FP       = 1.0, 1.0
MIN_CLUSTER      = 30      # below this many cases in a class, use the global threshold
LAMBDA_GRID      = [0.0, 1.0, 3.0, 6.0, 10.0, 20.0]
N_THRESHOLD_GRID = 250     # candidate thresholds swept per cluster
SEED             = 42

RUN_DIR = cfg.OUTPUT_DIR / TAG
FIG_DIR = cfg.FIGURE_DIR
TABLE_DIR = RUN_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

meta = json.loads((RUN_DIR / "meta.json").read_text())
CLUSTER_LABELS = meta["cluster_labels"]
K_CLUSTERS = meta["K_clusters"]
X_VOL_COLS = meta["x_vol_cols"]
N_CHAINS = meta["n_chains"]
CAL_YEAR, EV_YEAR = meta["calibration_year"], meta["evaluation_year"]

df_test = pd.read_csv(RUN_DIR / "test_frame.csv.gz")
bart = np.load(RUN_DIR / "bart.npz", allow_pickle=True)
p_bart = bart["p_test"]
bart_inclusion = bart["inclusion"]
bart_var_names = [str(v) for v in bart["var_names"]]

assert p_bart.shape[1] == len(df_test), "BART draws and the test frame disagree"
print(f"run '{TAG}': {meta['n_countries']} countries, "
      f"{len(df_test):,} test rows, {p_bart.shape[0]} BART draws")

# %%
# Draws

SCALARS = [
    "rho_global", "sigma_rho_m49", "tau_rho", "tau_em", "tau_at",
    "intercept_em", "intercept_at", "phi_disp_global", "tau_phi_disp",
    "mu_kappa", "sigma_kappa", "omega", "delta_phi",
]
VECTORS = {
    "beta_grav": [regions.pretty(c) for c in X_VOL_COLS],
    "theta_em": [r"$Z_{1}$"],
    "theta_at": [r"$Z_{1}$"],
    "rho_m49": CLUSTER_LABELS,
    "phi_disp_cluster": CLUSTER_LABELS,
    "kappa_m49": CLUSTER_LABELS,
}
SAMPLER = ["lp__", "divergent__", "treedepth__", "energy__", "stepsize__"]
PREDICTIVE = ["mu_dt_test", "phi_test"]

WANTED = SCALARS + list(VECTORS) + SAMPLER + PREDICTIVE

csv_files = [RUN_DIR / "stan" / name for name in meta["stan_csv"]]
with open(csv_files[0]) as handle:
    for line in handle:
        if not line.startswith("#"):
            header = line.strip().split(",")
            break

# Match on the exact name or the dotted prefix, so that 'rho_m49' does not also
# pull in 'rho_m49_lat' and 'alpha_em' does not pull in 'alpha_em_raw'.
keep = [c for c in header if any(c == w or c.startswith(w + ".") for w in WANTED)]
print(f"reading {len(keep):,} of {len(header):,} columns from {len(csv_files)} chains")

chains = []
for path in csv_files:
    chunk = pd.read_csv(path, comment="#", usecols=keep, engine="c")
    chains.append(chunk[keep].astype(np.float32))
draws = pd.concat(chains, ignore_index=True)
n_per_chain = len(chains[0])
assert all(len(c) == n_per_chain for c in chains), "chains have unequal lengths"
del chains
print(f"{len(draws):,} draws ({n_per_chain} per chain), "
      f"{draws.memory_usage().sum() / 1024 ** 2:.0f} MB")


def by_chain(column):
    """A column reshaped to (n_chains, n_draws) for the diagnostics."""
    return draws[column].to_numpy(dtype=float).reshape(N_CHAINS, n_per_chain)


def vector_columns(name):
    cols = [c for c in draws.columns if c.startswith(f"{name}.")]
    return sorted(cols, key=lambda c: int(c.split(".")[1]))


# %%
# Convergence

rows, missing = [], []
for name in SCALARS:
    if name not in draws.columns:
        missing.append(name)
        continue
    rows.append(diag.summarise(name, by_chain(name)))

for name, labels in VECTORS.items():
    cols = vector_columns(name)
    if not cols:
        missing.append(name)
        continue
    if len(cols) != len(labels):
        print(f"warning: {name} has {len(cols)} columns but {len(labels)} labels")
    for j, col in enumerate(cols):
        label = labels[j] if j < len(labels) else str(j + 1)
        rows.append(diag.summarise(f"{name}[{label}]", by_chain(col)))

if missing:
    print(f"warning: absent from the draws: {missing}")

convergence = pd.DataFrame(rows)
convergence.to_csv(TABLE_DIR / "convergence.csv", index=False)

n_divergent = int(draws["divergent__"].sum()) if "divergent__" in draws else 0
saturated = (float((draws["treedepth__"] >= meta["max_treedepth"]).mean())
             if "treedepth__" in draws else np.nan)
suspect = convergence[(convergence["rhat"] > 1.01) | (convergence["ess_bulk"] < 400)]

print(f"\ndivergent transitions: {n_divergent} "
      f"({n_divergent / len(draws) * 100:.2f}% of draws)")
print(f"treedepth saturated:   {saturated * 100:.1f}%")
print(f"worst R-hat: {convergence['rhat'].max():.4f} | "
      f"lowest bulk ESS: {convergence['ess_bulk'].min():.0f} | "
      f"parameters outside the thresholds: {len(suspect)}")
if len(suspect):
    print(suspect[["parameter", "rhat", "ess_bulk", "ess_tail"]]
          .sort_values("rhat", ascending=False).head(10).to_string(index=False))

# %%
# Posterior predictive draws of the volume component

mu_columns = vector_columns("mu_dt_test")
phi_columns = vector_columns("phi_test")
mu = draws[mu_columns].to_numpy(dtype=np.float32)
phi = draws[phi_columns].to_numpy(dtype=np.float32)
assert mu.shape[1] == len(df_test), "mu_dt_test and the test frame disagree"

# These two blocks are the bulk of the draws: 72,000 columns each at 190
# countries. Everything downstream reads them from the numpy copies, so give
# the DataFrame back its memory before simulating.
draws = draws.drop(columns=mu_columns + phi_columns)

finite = np.isfinite(mu).all(axis=1) & np.isfinite(phi).all(axis=1)
if not finite.all():
    print(f"dropping {(~finite).sum()} non-finite draws from the predictive step")
    mu, phi = mu[finite], phi[finite]

rng = np.random.default_rng(SEED)
flow_sim = ft.zero_truncated_negbin(mu, phi, rng)
flow_median = np.median(flow_sim, axis=0)
flow_q25 = np.percentile(flow_sim, 25, axis=0)
print(f"predictive draws: {flow_sim.shape[0]} x {flow_sim.shape[1]:,}")

# %%
# Threshold calibration.
#
# The hurdle returns a probability; turning it into an open/closed call needs a
# threshold, and the right threshold depends on what an error costs. Two rules
# are computed on the calibration year and applied unchanged to the evaluation
# year. The cost rule is the loss the paper argues for, but it needs a predicted
# volume and the volume model has seen the calibration year. The ROC rule uses
# the binary outcome only and is therefore immune to that overlap. They are
# reported side by side; the correlation between the two sets of thresholds is
# the robustness check.

is_cal = (df_test["is_evaluation"] == 0).to_numpy()
is_ev = (df_test["is_evaluation"] == 1).to_numpy()
df_ev = df_test.loc[is_ev].reset_index(drop=True)

p_cal = df_test.loc[is_cal, "p_hurdle"].to_numpy()
flow_cal = df_test.loc[is_cal, "flow"].to_numpy()
cluster_cal = df_test.loc[is_cal, "cluster"].to_numpy()
y_cal_bin = (flow_cal > 0).astype(int)

p_ev = df_ev["p_hurdle"].to_numpy()
cluster_ev = df_ev["cluster"].to_numpy()
y_true = df_ev["flow"].to_numpy()
y_true_bin = (y_true > 0).astype(int)

# Emerging corridors get the 25th percentile rather than the median: with no
# level to regress on, the predictive distribution is wide and right-skewed, and
# the median over-predicts under an absolute loss.
emergent_cal = df_test.loc[is_cal, "is_mig_lag"].fillna(0).to_numpy() == 0
emergent_ev = df_ev["is_mig_lag"].fillna(0).to_numpy() == 0
volume_cal = np.where(emergent_cal, flow_q25[is_cal], flow_median[is_cal])
volume_ev = np.where(emergent_ev, flow_q25[is_ev], flow_median[is_ev])

GRID = np.quantile(p_cal, np.linspace(0.05, 0.9995, N_THRESHOLD_GRID))


def thresholds_cost(lam):
    out = {}
    for c in np.unique(cluster_cal):
        m = cluster_cal == c
        f, v, p = flow_cal[m], volume_cal[m], p_cal[m]
        predicted = np.where(p[None, :] >= GRID[:, None], v[None, :], 0.0)
        err = np.abs(f[None, :] - predicted)
        loss = err.sum(axis=1) + lam * 100 * (err / (f[None, :] + 1)).sum(axis=1)
        out[int(c)] = float(GRID[int(np.argmin(loss))])
    return out


def thresholds_roc(w_fn=W_FN, w_fp=W_FP):
    fpr, tpr, thr = roc_curve(y_cal_bin, p_cal)
    fallback = float(thr[np.argmax(w_fn * tpr - w_fp * fpr)])
    out = {}
    for c in np.unique(cluster_cal):
        m = cluster_cal == c
        if (y_cal_bin[m].sum() < MIN_CLUSTER
                or (1 - y_cal_bin[m]).sum() < MIN_CLUSTER):
            out[int(c)] = fallback
            continue
        fpr, tpr, thr = roc_curve(y_cal_bin[m], p_cal[m])
        out[int(c)] = float(thr[np.argmax(w_fn * tpr - w_fp * fpr)])
    return out


def apply_thresholds(thr, p, cluster, flow, volume):
    default = float(np.median(list(thr.values())))
    cut = np.array([thr.get(int(k), default) for k in cluster])
    predicted_open = (p >= cut).astype(int)
    y_hat = np.where(predicted_open == 1, volume, 0.0)
    observed_open = (flow > 0).astype(int)
    err = np.abs(flow - y_hat)
    return {
        "FP": int(((predicted_open == 1) & (observed_open == 0)).sum()),
        "FN": int(((predicted_open == 0) & (observed_open == 1)).sum()),
        "TP": int(((predicted_open == 1) & (observed_open == 1)).sum()),
        "MAE": float(err.mean()),
        "MAPE": float((err / (flow + 1)).mean() * 100),
        "open": predicted_open,
    }


frontier = []
for lam in LAMBDA_GRID:
    thr = thresholds_cost(lam)
    cal = apply_thresholds(thr, p_cal, cluster_cal, flow_cal, volume_cal)
    ev = apply_thresholds(thr, p_ev, cluster_ev, y_true, volume_ev)
    frontier.append({
        "lambda": lam,
        "cal_FP": cal["FP"], "cal_FN": cal["FN"],
        "cal_MAE": cal["MAE"], "cal_MAPE": cal["MAPE"],
        "ev_FP": ev["FP"], "ev_FN": ev["FN"],
        "ev_MAE": ev["MAE"], "ev_MAPE": ev["MAPE"],
    })
frontier = pd.DataFrame(frontier)
frontier.to_csv(TABLE_DIR / "loss_frontier.csv", index=False)
print(f"\nloss frontier (thresholds fitted on {CAL_YEAR}, applied to {EV_YEAR}):")
print(frontier.round(1).to_string(index=False))

thr_cost = thresholds_cost(LAMBDA)
thr_roc = thresholds_roc()
shared = sorted(set(thr_cost) & set(thr_roc))
agreement = float(np.corrcoef([thr_cost[k] for k in shared],
                              [thr_roc[k] for k in shared])[0, 1])
print(f"\ncorrelation between the cost (lambda={LAMBDA}) and ROC thresholds: "
      f"{agreement:.3f}")
for name, thr in (("cost", thr_cost), ("roc", thr_roc)):
    r = apply_thresholds(thr, p_ev, cluster_ev, y_true, volume_ev)
    print(f"  {name:<5} FP {r['FP']:>6,}  FN {r['FN']:>6,}  "
          f"MAE {r['MAE']:>8,.0f}  MAPE {r['MAPE']:>6.1f}%")

thresholds = thr_cost if CALIBRATION_MODE == "cost" else thr_roc
result = apply_thresholds(thresholds, p_ev, cluster_ev, y_true, volume_ev)
y_pred_bin = result["open"]
y_pred = np.where(y_pred_bin == 1, volume_ev, 0.0)

pd.DataFrame({
    "cluster": [int(c) for c in sorted(thresholds)],
    "subregion": [CLUSTER_LABELS[int(c) - 1] for c in sorted(thresholds)],
    "threshold_cost": [thr_cost[c] for c in sorted(thresholds)],
    "threshold_roc": [thr_roc[c] for c in sorted(thresholds)],
}).to_csv(TABLE_DIR / "thresholds.csv", index=False)

# %%
# Predictive intervals: hurdle uncertainty times volume uncertainty.
#
# The interval has to compose both components. Taking the point probability and
# simulating a Bernoulli around it would hold the hurdle fixed and lose the term
# Var(E[flow | p]) of the variance decomposition, which is exactly what the
# paper claims is non-negligible.

sim_ev = flow_sim[:, is_ev]
n_volume_draws = sim_ev.shape[0]
pick = rng.choice(p_bart.shape[0], n_volume_draws,
                  replace=n_volume_draws > p_bart.shape[0])
p_draws_ev = p_bart[pick][:, is_ev]

open_sim = rng.binomial(1, np.clip(p_draws_ev, 0, 1))
flow_all = open_sim * sim_ev
pi_low = np.percentile(flow_all, 2.5, axis=0)
pi_high = np.percentile(flow_all, 97.5, axis=0)

# The same interval with the hurdle held at its posterior median, to size the
# contribution the composition adds.
open_plugin = rng.binomial(1, np.clip(np.tile(p_ev, (n_volume_draws, 1)), 0, 1))
plugin_all = open_plugin * sim_ev
pi_low_plugin = np.percentile(plugin_all, 2.5, axis=0)
pi_high_plugin = np.percentile(plugin_all, 97.5, axis=0)

covered = (y_true >= pi_low) & (y_true <= pi_high)
covered_plugin = (y_true >= pi_low_plugin) & (y_true <= pi_high_plugin)

# %%
# Headline metrics

abs_err = np.abs(y_true - y_pred)
metrics = {
    "n_dyads": int(len(y_true)),
    "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
    "precision": result["TP"] / max(result["TP"] + result["FP"], 1),
    "recall": result["TP"] / max(result["TP"] + result["FN"], 1),
    "auc": float(roc_auc_score(y_true_bin, p_ev)),
    "MAE": float(abs_err.mean()),
    "MAPE": float((abs_err / (y_true + 1)).mean() * 100),
    "WMAPE": float(abs_err.sum() / max(y_true.sum(), 1) * 100),
    "log_MAE": float(np.mean(np.abs(np.log1p(y_true) - np.log1p(y_pred)))),
    "coverage_95": float(covered.mean()),
    "coverage_95_plugin": float(covered_plugin.mean()),
    "mean_interval_width": float((pi_high - pi_low).mean()),
    "threshold_agreement": agreement,
    "divergent_transitions": n_divergent,
    "max_rhat": float(convergence["rhat"].max()),
    "min_ess_bulk": float(convergence["ess_bulk"].min()),
}
pd.Series(metrics).to_csv(TABLE_DIR / "metrics.csv", header=False)

print(f"\nout-of-sample {EV_YEAR}, {CALIBRATION_MODE} calibration")
print(f"  accuracy {metrics['accuracy'] * 100:5.1f}%   "
      f"precision {metrics['precision']:.3f}   recall {metrics['recall']:.3f}   "
      f"AUC {metrics['auc']:.4f}")
print(f"  MAE {metrics['MAE']:,.0f}   MAPE {metrics['MAPE']:.1f}%   "
      f"WMAPE {metrics['WMAPE']:.1f}%   log-MAE {metrics['log_MAE']:.4f}")
print(f"  95% coverage {metrics['coverage_95'] * 100:.1f}% "
      f"(hurdle held fixed: {metrics['coverage_95_plugin'] * 100:.1f}%)")

# Where the absolute error sits.
fp_mask = (y_pred_bin == 1) & (y_true_bin == 0)
fn_mask = (y_pred_bin == 0) & (y_true_bin == 1)
tp_mask = (y_pred_bin == 1) & (y_true_bin == 1)
print(f"  MAE decomposition: false positives {y_pred[fp_mask].sum() / len(y_true):,.1f}"
      f" | false negatives {y_true[fn_mask].sum() / len(y_true):,.1f}"
      f" | true positives {abs_err[tp_mask].sum() / len(y_true):,.1f}")
print(f"  the 20 worst dyads carry "
      f"{np.sort(abs_err)[-20:].sum() / abs_err.sum() * 100:.0f}% of the total error")

coverage_by_region = []
for c in sorted(np.unique(cluster_ev)):
    m = cluster_ev == c
    coverage_by_region.append({
        "cluster": int(c),
        "subregion": CLUSTER_LABELS[int(c) - 1],
        "n": int(m.sum()),
        "coverage": float(covered[m].mean()),
        "coverage_plugin": float(covered_plugin[m].mean()),
        "MAE": float(abs_err[m].mean()),
    })
coverage_by_region = pd.DataFrame(coverage_by_region)
coverage_by_region.to_csv(TABLE_DIR / "coverage_by_subregion.csv", index=False)

# %%
# Figures

print(f"\nfigures -> {FIG_DIR}")

# Volume coefficients.
beta = draws[vector_columns("beta_grav")].to_numpy(dtype=float)
fig, ax = fs.figure(height=0.30 * len(X_VOL_COLS) + 0.85)
fs.interval_plot(ax, beta, labels=[regions.pretty(c) for c in X_VOL_COLS])
ax.set_xlabel("Posterior coefficient, standardised covariates")
fs.save(fig, "fig_gravity_coefficients", FIG_DIR)

# %%
# Sub-regional heterogeneity. Both panels use the ordering of the first, so
# that a sub-region sits on the same row in each and the two components can be
# read against one another.
rho = draws[vector_columns("rho_m49")].to_numpy(dtype=float)
phi_cluster = draws[vector_columns("phi_disp_cluster")].to_numpy(dtype=float)

fig, axes = plt.subplots(
    1, 2, figsize=(fs.TEXT_WIDTH, 0.30 * K_CLUSTERS + 1.0), sharey=True,
    layout="constrained")
order = fs.halfeye(axes[0], rho, labels=CLUSTER_LABELS, sort_by="median",
                   color=fs.BLUE)
axes[0].set_xlabel(r"AR(1) inertia $\rho_k$")
fs.halfeye(axes[1], phi_cluster, labels=CLUSTER_LABELS, sort_by=order,
           color=fs.VERMILION)
axes[1].set_xlabel(r"Dispersion $\phi_k$ (lower is more dispersed)")
axes[1].tick_params(axis="y", labelleft=False)
fs.save(fig, "fig_subregion_heterogeneity", FIG_DIR)

# %%
# BART variable inclusion: the share of splitting rules that use each
# covariate, per posterior draw. With the tree count held at 200 this is the
# standard BART read on which covariates the hurdle actually leans.
inclusion = bart_inclusion / np.clip(bart_inclusion.sum(axis=1, keepdims=True), 1, None)
keep_top = np.argsort(np.median(inclusion, axis=0))[::-1][:18]
fig, ax = fs.figure(height=0.27 * len(keep_top) + 0.85)
fs.interval_plot(ax, inclusion[:, keep_top],
                 labels=[regions.pretty(bart_var_names[j]) for j in keep_top],
                 color=fs.GREEN, zero_line=False)
ax.set_xlabel("Share of splitting rules")
ax.set_xlim(left=0)
fs.save(fig, "fig_bart_inclusion", FIG_DIR)

# %%
# Hurdle discrimination and calibration.
fig, axes = plt.subplots(1, 2, figsize=(fs.TEXT_WIDTH, 2.9), layout="constrained")

ax = axes[0]
for label, mask, colour in ((f"{CAL_YEAR}", is_cal, fs.MUTED),
                            (f"{EV_YEAR}", is_ev, fs.BLUE)):
    y = (df_test.loc[mask, "flow"] > 0).astype(int)
    p = df_test.loc[mask, "p_hurdle"]
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, color=colour, lw=1.1,
            label=f"{label} (AUC {roc_auc_score(y, p):.3f})")
ax.plot([0, 1], [0, 1], color=fs.RULE, lw=0.6, zorder=0)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.legend(loc="lower right")

# Reliability: a probability of 0.3 should be right three times in ten.
ax = axes[1]
edges = np.unique(np.quantile(p_ev, np.linspace(0, 1, 11)))
bin_id = np.clip(np.digitize(p_ev, edges[1:-1]), 0, len(edges) - 2)
mean_p, obs, n_bin = [], [], []
for b in range(len(edges) - 1):
    m = bin_id == b
    if m.sum() < 10:
        continue
    mean_p.append(p_ev[m].mean())
    obs.append(y_true_bin[m].mean())
    n_bin.append(m.sum())
mean_p, obs, n_bin = np.array(mean_p), np.array(obs), np.array(n_bin)
lo, hi = fs.wilson(obs * n_bin, n_bin)
ax.plot([0, 1], [0, 1], color=fs.RULE, lw=0.6, zorder=0)
ax.vlines(mean_p, lo, hi, color=fs.BLUE, lw=0.8)
ax.plot(mean_p, obs, "o", color="white", mec=fs.BLUE, mew=0.9, ms=4)
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
fs.save(fig, "fig_hurdle_discrimination", FIG_DIR)

# %%
# Out-of-sample predictions.
fig, axes = plt.subplots(1, 2, figsize=(fs.TEXT_WIDTH, 2.9), layout="constrained")

ax = axes[0]
both_positive = (y_true > 0) & (y_pred > 0)
hb = ax.hexbin(np.log10(y_true[both_positive]), np.log10(y_pred[both_positive]),
               gridsize=42, bins="log", mincnt=1, linewidths=0,
               cmap="Blues", rasterized=True)
limits = [0, np.log10(max(y_true.max(), y_pred.max())) * 1.02]
ax.plot(limits, limits, color=fs.INK, lw=0.7, ls=(0, (3, 2)), zorder=3)
ax.set_xlim(limits)
ax.set_ylim(limits)
ax.set_aspect("equal")
ax.set_xlabel(r"Observed flow, $\log_{10}$")
ax.set_ylabel(r"Predicted flow, $\log_{10}$")
bar = fig.colorbar(hb, ax=ax, pad=0.015, fraction=0.045, aspect=28)
bar.set_label("Dyads", labelpad=1)
bar.outline.set_linewidth(0.4)
bar.ax.tick_params(labelsize=7, width=0.5, length=2)

# Error concentration: how much of the total absolute error a small number of
# corridors accounts for.
ax = axes[1]
sorted_err = np.sort(abs_err)[::-1]
share_dyads = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
share_error = np.cumsum(sorted_err) / sorted_err.sum()
ax.plot(share_dyads * 100, share_error * 100, color=fs.VERMILION, lw=1.1)
ax.plot([0, 100], [0, 100], color=fs.RULE, lw=0.6, zorder=0)
for mark in (0.001, 0.01, 0.1):
    idx = max(int(mark * len(sorted_err)) - 1, 0)
    ax.plot([mark * 100], [share_error[idx] * 100], "o", color="white",
            mec=fs.VERMILION, mew=0.9, ms=4, zorder=4)
    ax.annotate(fs.pct(f"{share_error[idx] * 100:.0f}%"),
                (mark * 100, share_error[idx] * 100), textcoords="offset points",
                xytext=(5, -7), fontsize=7, color=fs.INK)
ax.set_xscale("log")
ax.set_xlabel(fs.pct("Share of dyads, worst first (%)"))
ax.set_ylabel(fs.pct("Cumulative share of error (%)"))
ax.set_ylim(0, 101)
ax.yaxis.grid(True, color=fs.RULE, lw=0.4)
fs.save(fig, "fig_predictive_scatter", FIG_DIR)

# %%
# Interval coverage.
fig, axes = plt.subplots(1, 2, figsize=(fs.TEXT_WIDTH, 0.26 * K_CLUSTERS + 1.3),
                         gridspec_kw={"width_ratios": [1.35, 1]},
                         layout="constrained")

ax = axes[0]
tab = coverage_by_region.sort_values("coverage").reset_index(drop=True)
lo, hi = fs.wilson(tab["coverage"] * tab["n"], tab["n"])
rows = np.arange(len(tab))
fs.row_bands(ax, len(tab))
ax.axvline(0.95, color=fs.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
ax.hlines(rows, lo, hi, color=fs.BLUE, lw=0.9, zorder=2)
ax.plot(tab["coverage"], rows, "o", color="white", mec=fs.BLUE, mew=0.9,
        ms=4, zorder=3)
ax.set_yticks(rows)
ax.set_yticklabels(tab["subregion"])
ax.set_ylim(-0.6, len(tab) - 0.4)
ax.invert_yaxis()
ax.set_xlabel(fs.pct("Empirical coverage of the 95% interval"))
fs.despine(ax, left=True)
ax.xaxis.grid(True, color=fs.RULE, lw=0.4)

# Coverage against the size of the observed flow, which is where a hurdle model
# is most likely to be miscalibrated.
ax = axes[1]
positive = y_true > 0
deciles = np.unique(np.quantile(y_true[positive], np.linspace(0, 1, 9)))
centres, values, counts = [], [], []
for b in range(len(deciles) - 1):
    m = positive & (y_true >= deciles[b]) & (y_true < deciles[b + 1])
    if m.sum() < 20:
        continue
    centres.append(np.sqrt(max(deciles[b], 1) * deciles[b + 1]))
    values.append(covered[m].mean())
    counts.append(m.sum())
centres, values, counts = np.array(centres), np.array(values), np.array(counts)
lo, hi = fs.wilson(values * counts, counts)
ax.axhline(0.95, color=fs.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
zero_cov = covered[~positive].mean()
ax.axhline(zero_cov, color=fs.GREEN, lw=0.8, zorder=1)
ax.annotate("closed corridors", (centres[0], zero_cov), textcoords="offset points",
            xytext=(0, 4), fontsize=7, color=fs.GREEN)
ax.vlines(centres, lo, hi, color=fs.VERMILION, lw=0.9)
ax.plot(centres, values, "o-", color=fs.VERMILION, mfc="white", mew=0.9, ms=4,
        lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("Observed flow")
ax.set_ylabel("Empirical coverage")
ax.set_ylim(0, 1.06)
ax.yaxis.grid(True, color=fs.RULE, lw=0.4)
fs.save(fig, "fig_coverage", FIG_DIR)

# %%
# Loss frontier.
fig, axes = plt.subplots(1, 2, figsize=(fs.TEXT_WIDTH, 2.7), layout="constrained")

ax = axes[0]
ax.plot(frontier["ev_MAE"], frontier["ev_MAPE"], "-", color=fs.BLUE, lw=0.9,
        zorder=2)
ax.plot(frontier["ev_MAE"], frontier["ev_MAPE"], "o", color="white",
        mec=fs.BLUE, mew=0.9, ms=4, zorder=3)
chosen = frontier.loc[(frontier["lambda"] - LAMBDA).abs().idxmin()]
ax.plot([chosen["ev_MAE"]], [chosen["ev_MAPE"]], "o", color=fs.VERMILION,
        ms=5, zorder=4)
ax.set_xlabel("Mean absolute error")
ax.set_ylabel(fs.pct("Mean absolute percentage error (%)"))
ax.margins(x=0.16, y=0.16)
ax.grid(True, color=fs.RULE, lw=0.4)
fs.thousands(ax, axis="x")
# After the limits are settled, so that the display-space normals are right.
fs.label_points(ax, frontier["ev_MAE"], frontier["ev_MAPE"],
                [rf"$\lambda={v:.0f}$" for v in frontier["lambda"]])

ax = axes[1]
ax.plot(frontier["lambda"], frontier["ev_FN"], "o-", color=fs.VERMILION,
        mfc="white", mew=0.9, ms=4, lw=0.9, label="False negatives")
ax.plot(frontier["lambda"], frontier["ev_FP"], "s-", color=fs.BLUE,
        mfc="white", mew=0.9, ms=4, lw=0.9, label="False positives")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel("Dyads")
ax.grid(True, color=fs.RULE, lw=0.4)
ax.legend(loc="best")
fs.thousands(ax, axis="y")
fs.save(fig, "fig_loss_frontier", FIG_DIR)

# %%
# Where the classification fails.
df_ev = df_ev.assign(false_negative=fn_mask.astype(int),
                     false_positive=fp_mask.astype(int))
by_origin = (df_ev.groupby("orig")[["false_negative", "false_positive"]]
             .sum()
             .assign(total=lambda d: d["false_negative"] + d["false_positive"])
             .sort_values("total", ascending=False)
             .head(22)
             .sort_values("total"))

fig, ax = fs.figure(height=0.26 * len(by_origin) + 0.9)
rows = np.arange(len(by_origin))
fs.row_bands(ax, len(by_origin))
ax.barh(rows, -by_origin["false_negative"], height=0.62, color=fs.VERMILION,
        edgecolor="none", zorder=2)
ax.barh(rows, by_origin["false_positive"], height=0.62, color=fs.BLUE,
        edgecolor="none", zorder=2)
ax.axvline(0, color=fs.INK, lw=0.6, zorder=3)
ax.set_yticks(rows)
ax.set_yticklabels(by_origin.index)
ax.set_ylim(-0.6, len(by_origin) - 0.4)
# Two labels rather than one, each in the colour of the bars it names, so the
# reader never has to work out which side is which.
ax.text(0.25, -0.055, "Missed openings", transform=ax.transAxes, ha="center",
        va="top", color=fs.VERMILION, fontsize=8)
ax.text(0.75, -0.055, "Spurious openings", transform=ax.transAxes, ha="center",
        va="top", color=fs.BLUE, fontsize=8)
fs.despine(ax, left=True)
ax.xaxis.grid(True, color=fs.RULE, lw=0.4)
ax.xaxis.set_major_formatter(plt.FuncFormatter(
    lambda v, _p: f"{abs(v):,.0f}".replace(",", r"\," if fs._USE_TEX else " ")))
fs.save(fig, "fig_error_by_origin", FIG_DIR)

# %%
# Convergence.
fig, axes = plt.subplots(1, 2, figsize=(fs.TEXT_WIDTH, 2.4), layout="constrained")

ax = axes[0]
values = convergence["rhat"].dropna().sort_values().to_numpy()
ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post",
        color=fs.BLUE, lw=1.1)
ax.axvline(1.01, color=fs.VERMILION, lw=0.8, ls=(0, (3, 2)))
ax.set_xlabel(r"$\widehat{R}$")
ax.set_ylabel("Cumulative share of parameters")
ax.set_ylim(0, 1.02)
ax.grid(True, color=fs.RULE, lw=0.4)

ax = axes[1]
ratio_bulk = convergence["ess_bulk"] / len(draws)
ratio_tail = convergence["ess_tail"] / len(draws)
ax.scatter(ratio_bulk, ratio_tail, s=9, color=fs.BLUE, edgecolors="none",
           alpha=0.8)
lim = float(np.nanmax([ratio_bulk.max(), ratio_tail.max()])) * 1.05
ax.plot([0, lim], [0, lim], color=fs.RULE, lw=0.6, zorder=0)
ax.axvline(400 / len(draws), color=fs.VERMILION, lw=0.8, ls=(0, (3, 2)))
ax.axhline(400 / len(draws), color=fs.VERMILION, lw=0.8, ls=(0, (3, 2)))
ax.set_xlabel("Bulk ESS / draws")
ax.set_ylabel("Tail ESS / draws")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.grid(True, color=fs.RULE, lw=0.4)
fs.save(fig, "fig_convergence", FIG_DIR)

print(f"\ntables -> {TABLE_DIR}")
print("done")
