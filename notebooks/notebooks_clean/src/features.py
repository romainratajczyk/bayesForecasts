"""Feature construction and the design matrices handed to Stan and to BART."""


import numpy as np
import pandas as pd


def add_log_covariates(df, raw_columns):
    """Log the gravity masses. Zeros become missing rather than -inf; the rows
    are dropped later by the explicit dropna on the required column list."""
    out = {}
    for raw in raw_columns:
        out[f"log_{raw}"] = np.log(df[raw].replace(0, np.nan))
    return df.assign(**out)


def add_flow_momentum(df):
    """Momentum: the change in log flow over the previous two waves.

    The AR(1) term carries the level at t-1; this carries the slope, which is
    what separates a corridor at 10,000 and falling from one at 10,000 and
    rising. Built from the raw lags, not from the zero-filled version, so that
    a closed corridor contributes no spurious trend. Dyads without two
    observed waves get zero, meaning "no trend observed".
    """
    df = df.sort_values(["dyad", "year"])
    lag1 = df.groupby("dyad")["log_flow"].shift(1)
    lag2 = df.groupby("dyad")["log_flow"].shift(2)
    df["flow_momentum"] = (lag1 - lag2).fillna(0.0)
    return df.sort_values(["orig", "dest", "year"]).reset_index(drop=True)


def add_degree_features(df_hurdle, df_test, df_volume):
    """Network position of each country at t-1.

    out_degree_o and in_degree_d count the corridors open at t-1 out of the
    origin and into the destination; their product is the crude prediction a
    configuration model would make for the dyad. The test frame inherits the
    training-period averages rather than its own contemporaneous degrees,
    which keeps 2015 information out of a 2015 covariate.
    """
    out_deg = (df_hurdle.groupby(["orig", "year"])["is_mig_lag"].sum()
               .reset_index(name="out_degree_o"))
    in_deg = (df_hurdle.groupby(["dest", "year"])["is_mig_lag"].sum()
              .reset_index(name="in_degree_d"))

    df_hurdle = df_hurdle.merge(out_deg, on=["orig", "year"], how="left")
    df_hurdle = df_hurdle.merge(in_deg, on=["dest", "year"], how="left")
    df_hurdle["transitivity_proxy"] = (
        df_hurdle["out_degree_o"].fillna(0) * df_hurdle["in_degree_d"].fillna(0)
    )

    out_mean = df_hurdle.groupby("orig")["out_degree_o"].mean().reset_index()
    in_mean = df_hurdle.groupby("dest")["in_degree_d"].mean().reset_index()
    df_test = df_test.merge(out_mean, on="orig", how="left")
    df_test = df_test.merge(in_mean, on="dest", how="left")
    df_test["transitivity_proxy"] = (
        df_test["out_degree_o"].fillna(0) * df_test["in_degree_d"].fillna(0)
    )

    countries = sorted(
        set(df_hurdle["orig"]) | set(df_hurdle["dest"])
        | set(df_test["orig"]) | set(df_test["dest"])
        | set(df_volume["orig"]) | set(df_volume["dest"])
    )
    for frame in (df_hurdle, df_test, df_volume):
        frame["A2_log"] = np.log1p(_two_step_paths(frame, countries))

    return df_hurdle, df_test, df_volume


def _two_step_paths(frame, countries):
    """Number of active two-step routes i -> k -> j at t-1.

    With A[i, j] = 1 if the corridor was open at t-1, (A @ A)[i, j] counts the
    intermediaries k through which i already reaches j. This is the one signal
    that is non-zero precisely where every dyadic inertia covariate is silent,
    which is where the false negatives concentrate.
    """
    index = {c: i for i, c in enumerate(countries)}
    n = len(countries)
    out = np.zeros(len(frame))
    for _, sub in frame.groupby("year"):
        A = np.zeros((n, n), dtype=np.float32)
        active = sub[sub["is_mig_lag"] == 1]
        A[active["orig"].map(index).values, active["dest"].map(index).values] = 1.0
        A2 = A @ A
        rows = frame.index.get_indexer(sub.index)
        out[rows] = A2[sub["orig"].map(index).values, sub["dest"].map(index).values]
    return out


def standardize(X, columns, binary_columns, stats=None):
    """Centre and scale, leaving indicator columns alone.

    Pass the training `stats` back in when transforming the test matrix, so
    that the test years are never used to compute a mean.
    """
    X = np.asarray(X, dtype=float)
    out = X.copy()
    fitted = {}
    for j, col in enumerate(columns):
        if col in binary_columns:
            fitted[col] = {"mean": 0.0, "std": 1.0}
            continue
        if stats is None:
            mu, sd = X[:, j].mean(), max(X[:, j].std(), 1e-8)
        else:
            mu, sd = stats[col]["mean"], stats[col]["std"]
        out[:, j] = (X[:, j] - mu) / sd
        fitted[col] = {"mean": float(mu), "std": float(sd)}
    return out, fitted


def build_country_covariate(df_last, country_to_id, iso3_to_m49):
    """The single country-level predictor Z of the emission and attraction
    effects: log population plus log GDP per capita at the last training wave,
    i.e. a log economic mass.

    Countries absent from the last wave inherit the median of their M49
    sub-region, and failing that the global median. Standardised at the end so
    that the hyper-regression coefficients are on a comparable scale.
    """
    n_countries = len(country_to_id)
    Z = np.full((n_countries, 1), np.nan)

    for country, cid in country_to_id.items():
        as_origin = df_last[df_last["orig"] == country]
        as_dest = df_last[df_last["dest"] == country]
        if not as_origin.empty:
            mass = (as_origin["log_P_it"].iloc[0]
                    + as_origin["log_gdpcap_o_lag1"].iloc[0])
        elif not as_dest.empty:
            mass = (as_dest["log_P_jt"].iloc[0]
                    + as_dest["log_gdpcap_d_lag1"].iloc[0])
        else:
            mass = np.nan
        Z[cid - 1, 0] = mass

    global_median = np.nanmedian(Z[:, 0])
    for country, cid in country_to_id.items():
        if not np.isnan(Z[cid - 1, 0]):
            continue
        region = iso3_to_m49.get(country, 99)
        peers = [Z[country_to_id[p] - 1, 0]
                 for p, m in iso3_to_m49.items()
                 if m == region and p in country_to_id
                 and not np.isnan(Z[country_to_id[p] - 1, 0])]
        Z[cid - 1, 0] = np.median(peers) if peers else global_median

    Z[:, 0] = (Z[:, 0] - Z[:, 0].mean()) / Z[:, 0].std()
    return Z


def markov_states(df):
    """Split the positive flows into the three regimes the volume model treats
    differently.

    continuing : open at t-1, so the AR(1) term applies.
    virgin     : never open before, so there is no level to regress on and the
                 cluster intercept kappa takes over.
    reopening  : closed at t-1 but open at some earlier wave. The AR term has
                 no meaning here and the corridor is not new either, so these
                 rows are dropped from the likelihood rather than forced into
                 one of the two branches.
    """
    has_history = (
        df.groupby(["orig", "dest"])["flow"]
        .transform(lambda x: (x.shift(1) > 0).expanding().max()) > 0
    ).astype(int)
    continuing = df["is_mig_lag"] == 1
    virgin = (df["is_mig_lag"] == 0) & (has_history == 0)
    reopening = (df["is_mig_lag"] == 0) & (has_history == 1)
    return has_history, continuing, virgin, reopening


def check_finite(stan_data):
    """Stan gives an unhelpful error on a NaN buried in a 70,000-row matrix."""
    bad = []
    for key, value in stan_data.items():
        if not isinstance(value, (list, np.ndarray)):
            continue
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number) and not np.isfinite(arr).all():
            bad.append(key)
    if bad:
        raise ValueError(f"non-finite values in stan_data: {bad}")
    return True


def zero_truncated_negbin(mu, phi, rng, max_retries=50, block=4096):
    """Draw from the zero-truncated negative binomial by rejection.

    Stan reports the untruncated mean and dispersion; the likelihood is
    truncated, so the predictive draws have to be too. Redrawing until the
    value is positive is exactly the truncated distribution. The handful of
    cells where the success probability is so small that fifty attempts all
    return zero are set to one, which is the mode of the truncated law there.

    Done in column blocks and written into an int32 result: at 190 countries
    this array is 1,600 draws by 72,000 dyads, and numpy's default int64
    intermediates would cost a gigabyte apiece.
    """
    mu = np.asarray(mu)
    phi = np.asarray(phi)
    n_draws, n_columns = mu.shape
    out = np.empty((n_draws, n_columns), dtype=np.int32)

    for start in range(0, n_columns, block):
        stop = min(start + block, n_columns)
        eta = np.clip(mu[:, start:stop], -50.0, 50.0)
        size = np.clip(phi[:, start:stop], 1e-8, 1e8)
        p = np.clip(size / (size + np.exp(eta)), 1e-10, 1.0 - 1e-10)

        chunk = rng.negative_binomial(size, p)
        zeros = chunk == 0
        for _ in range(max_retries):
            if not zeros.any():
                break
            chunk[zeros] = rng.negative_binomial(size[zeros], p[zeros])
            zeros = chunk == 0
        chunk[zeros] = 1
        out[:, start:stop] = chunk

    return out


def frame_signature(df, columns):
    """Cheap guard against the frames drifting apart between the two scripts."""
    return {
        "n_rows": int(len(df)),
        "columns": list(columns),
        "flow_sum": float(pd.to_numeric(df["flow"], errors="coerce").sum()),
    }
