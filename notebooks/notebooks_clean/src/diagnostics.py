"""MCMC convergence diagnostics, following Vehtari et al. (2021).

The notebook this replaces estimated the effective sample size from the lag-1
autocorrelation of the four chains concatenated end to end, which both ignores
the autocorrelation past lag 1 and treats the joins between chains as if they
were part of the sequence. What follows is the published estimator: rank
normalisation, split chains, folded R-hat, and Geyer's initial monotone
positive sequence for the autocorrelation sum. It depends on numpy only, so a
reader can run it without installing ArviZ.

Every function takes draws shaped (n_chains, n_draws).
"""


import numpy as np
from scipy.special import ndtri


def _split(draws):
    """Split each chain in half. Doubles the number of sequences compared, so
    that a chain drifting slowly within itself is caught."""
    n_chains, n_draws = draws.shape
    half = n_draws // 2
    return np.concatenate([draws[:, :half], draws[:, half:2 * half]], axis=0)


def _rank_normalise(draws):
    """Map the pooled draws to normal scores. Makes R-hat well defined for
    parameters with no finite variance, and insensitive to the scale."""
    shape = draws.shape
    flat = draws.reshape(-1)
    order = flat.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, flat.size + 1)
    # Average ranks over ties, otherwise duplicated draws bias the scores.
    _, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sums = np.bincount(inverse, weights=ranks)
        ranks = (sums / counts)[inverse]
    z = ndtri((ranks - 3.0 / 8.0) / (flat.size - 0.25 + 0.5))
    return z.reshape(shape)


def _rhat_plain(draws):
    n_chains, n_draws = draws.shape
    if n_draws < 2 or n_chains < 2:
        return np.nan
    chain_means = draws.mean(axis=1)
    chain_vars = draws.var(axis=1, ddof=1)
    W = chain_vars.mean()
    if W <= 0 or not np.isfinite(W):
        return np.nan
    B = n_draws * chain_means.var(ddof=1)
    var_hat = (n_draws - 1) / n_draws * W + B / n_draws
    return float(np.sqrt(var_hat / W))


def rhat(draws):
    """Rank-normalised split R-hat, maximised against its folded counterpart.

    The fold catches chains that agree on the centre but disagree on the
    spread, which the plain statistic is blind to.
    """
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2:
        raise ValueError("draws must be (n_chains, n_draws)")
    if np.allclose(draws, draws.flat[0]):
        return np.nan

    split = _split(draws)
    bulk = _rhat_plain(_rank_normalise(split))
    folded = _rhat_plain(_rank_normalise(np.abs(split - np.median(split))))
    return float(np.nanmax([bulk, folded]))


def _autocov(x):
    """Autocovariance of one chain by FFT, at every lag."""
    n = x.size
    x = x - x.mean()
    n_fft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n_fft)
    acov = np.fft.irfft(f * np.conjugate(f), n_fft)[:n]
    return acov / n


def _ess_from(draws):
    """ESS of already-transformed draws, with Geyer's positive sequence."""
    n_chains, n_draws = draws.shape
    if n_draws < 4:
        return np.nan

    acov = np.array([_autocov(draws[c]) for c in range(n_chains)])
    chain_mean_var = acov[:, 0] * n_draws / (n_draws - 1)
    mean_var = chain_mean_var.mean()
    if mean_var <= 0 or not np.isfinite(mean_var):
        return np.nan

    var_plus = mean_var * (n_draws - 1) / n_draws
    if n_chains > 1:
        var_plus += draws.mean(axis=1).var(ddof=1)

    rho = np.ones(n_draws)
    rho[1:] = 1.0 - (mean_var - acov[:, 1:].mean(axis=0)) / var_plus

    # Geyer's initial positive sequence: sum the autocorrelations in adjacent
    # pairs, starting at (rho_0, rho_1), and stop at the first pair that turns
    # negative. Then enforce the monotone decrease that must hold for a
    # reversible chain, which trims the noise in the far lags.
    n_pairs = n_draws // 2
    pairs = rho[0:2 * n_pairs:2] + rho[1:2 * n_pairs:2]
    negative = np.nonzero(pairs < 0)[0]
    pairs = pairs[:negative[0]] if negative.size else pairs
    if pairs.size:
        pairs = np.maximum(np.minimum.accumulate(pairs), 0.0)

    tau = -1.0 + 2.0 * float(pairs.sum())
    tau = max(tau, 1.0 / np.log10(max(n_chains * n_draws, 11)))
    return float(n_chains * n_draws / tau)


def ess_bulk(draws):
    """Effective sample size for the centre of the distribution."""
    draws = np.asarray(draws, dtype=float)
    if np.allclose(draws, draws.flat[0]):
        return np.nan
    return _ess_from(_rank_normalise(_split(draws)))


def ess_tail(draws):
    """Effective sample size for the 5% and 95% quantiles.

    Reported alongside the bulk value because the credible bounds in the
    coefficient figures are tail quantities: a healthy bulk ESS says nothing
    about how well resolved the ends of the interval are.
    """
    draws = np.asarray(draws, dtype=float)
    if np.allclose(draws, draws.flat[0]):
        return np.nan
    q05, q95 = np.percentile(draws, [5, 95])
    below = _ess_from(_split((draws <= q05).astype(float)))
    above = _ess_from(_split((draws >= q95).astype(float)))
    return float(np.nanmin([below, above]))


def mcse_mean(draws):
    """Monte Carlo standard error of the posterior mean."""
    draws = np.asarray(draws, dtype=float)
    ess = ess_bulk(draws)
    if not np.isfinite(ess) or ess <= 0:
        return np.nan
    return float(draws.std(ddof=1) / np.sqrt(ess))


def summarise(name, draws):
    """One row of the convergence table."""
    draws = np.asarray(draws, dtype=float)
    pooled = draws.reshape(-1)
    q = np.percentile(pooled, [2.5, 25, 50, 75, 97.5])
    return {
        "parameter": name,
        "mean": pooled.mean(),
        "sd": pooled.std(ddof=1),
        "q2.5": q[0],
        "q25": q[1],
        "median": q[2],
        "q75": q[3],
        "q97.5": q[4],
        "mcse": mcse_mean(draws),
        "ess_bulk": ess_bulk(draws),
        "ess_tail": ess_tail(draws),
        "rhat": rhat(draws),
    }
