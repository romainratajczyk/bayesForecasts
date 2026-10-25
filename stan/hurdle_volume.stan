// Volume part of the hurdle model: zero-truncated NB2 for positive flows. Mean: country effects with the GDP of
// the period, gravity covariates, an AR term in deviations from the lagged structural mean (two-lag average with
// weight omega), and iid country-period shocks. Persistence, entry level and dispersion pooled by M49 sub-region
// of origin.
data {
  int<lower=1> N_c;                     // countries
  int<lower=1> P;                       // training periods 1990, ..., T - 5
  int<lower=1> M;                       // M49 sub-regions of the origin
  int<lower=1> K;                       // covariates in X
  matrix[N_c, P + 1] Z;                 // standardised log GDP of year t - 1; column P + 1 is the forecast period T
  vector[K] beta_mu;                    // prior of beta, set by covariate name
  vector<lower=0>[K] beta_sd;

  // cells (dyad, period) whose structural mean is needed; the first N_cell_tr serve training
  int<lower=1> N_cell;
  int<lower=1, upper=N_cell> N_cell_tr;
  array[N_cell] int<lower=1> li_o;      // (period - 1) * N_c + origin
  array[N_cell] int<lower=1> li_d;      // (period - 1) * N_c + destination
  matrix[N_cell, K] X;

  // training observations (positive flows)
  int<lower=1> N;
  int<lower=1> D;
  array[N] int<lower=1> y;
  array[N] int<lower=1, upper=D> dyad;
  array[D] int<lower=1, upper=M> cl;
  array[N] int<lower=1, upper=N_cell_tr> c0;   // cell at t
  array[N] int<lower=1, upper=N_cell_tr> c1;   // cell at t - 5 (c0 if unavailable)
  array[N] int<lower=1, upper=N_cell_tr> c2;   // cell at t - 10 (c1 if unavailable)
  vector<lower=0, upper=1>[N] is_ar;           // Y_{t-5} > 0
  vector<lower=0, upper=1>[N] has_l2;          // Y_{t-10} > 0 and its cell available
  vector[N] ly1;                               // log Y_{t-5} (0 if Y_{t-5} = 0)
  vector[N] mom;                               // log Y_{t-5} - log Y_{t-10} (0 if has_l2 = 0)
  vector[N] s;                                 // standardised log Y_{t-5}

  // country-period shocks: Q training periods with observations, then T
  int<lower=1> Q;
  array[N] int<lower=1, upper=N_c * Q> xo;     // (period - 1) * N_c + origin
  array[N] int<lower=1, upper=N_c * Q> xd;     // (period - 1) * N_c + destination

  // forecast rows (calibration years and T)
  int<lower=0> N_t;
  array[N_t] int<lower=0, upper=D> dyad_t;     // 0: dyad unseen in training
  array[N_t] int<lower=1, upper=M> cl_t;
  array[N_t] int<lower=1, upper=N_cell> c0_t;
  array[N_t] int<lower=1, upper=N_cell> c1_t;
  array[N_t] int<lower=1, upper=N_cell> c2_t;
  vector<lower=0, upper=1>[N_t] is_ar_t;
  vector<lower=0, upper=1>[N_t] has_l2_t;
  vector[N_t] ly1_t;
  vector[N_t] mom_t;
  vector[N_t] s_t;
  array[N_t] int<lower=1, upper=N_c * (Q + 1)> xo_t;   // period Q + 1 is T
  array[N_t] int<lower=1, upper=N_c * (Q + 1)> xd_t;
}
parameters {
  real c;                               // intercept: only the sum of the origin and destination intercepts is identified
  real theta_em;
  real theta_at;
  real<lower=0> tau_em;
  real<lower=0> tau_at;
  vector[N_c] a_em;
  vector[N_c] a_at;
  real<lower=0> sig_xi_em;
  real<lower=0> sig_xi_at;
  vector[N_c * Q] xi_em;                // centred: each shock is informed by the flows of its country and period
  vector[N_c * Q] xi_at;

  vector[K] beta;

  real mu_r;                            // persistence: global -> M49 -> dyad, on the atanh scale
  real<lower=0> sigma_r;
  vector[M] r_raw;
  real<lower=0> tau_rho;
  vector[D] eps;

  real mu_kappa;
  real<lower=0> sigma_kappa;
  vector[M] kappa_raw;

  real omega_raw;

  real<lower=0> phi_global;
  vector<lower=0>[M] phi_k;
  real<lower=0> tau_phi;
  vector[D] eps2;
  real delta_phi;
}
transformed parameters {
  real<lower=0, upper=1> omega = inv_logit(omega_raw);
  vector[M] r = mu_r + sigma_r * r_raw;
  vector[M] kappa = mu_kappa + sigma_kappa * kappa_raw;
  matrix[N_c, P] alpha = rep_matrix(c + tau_em * a_em, P) + theta_em * Z[:, 1:P];
  matrix[N_c, P] gamma = rep_matrix(tau_at * a_at, P) + theta_at * Z[:, 1:P];
}
model {
  c ~ normal(0, sqrt2());
  theta_em ~ normal(0, 0.5);
  theta_at ~ normal(0, 0.5);
  tau_em ~ normal(0, 0.25);
  tau_at ~ normal(0, 0.25);
  a_em ~ std_normal();
  a_at ~ std_normal();
  sig_xi_em ~ normal(0, 0.5);
  sig_xi_at ~ normal(0, 0.5);
  xi_em ~ normal(0, sig_xi_em);
  xi_at ~ normal(0, sig_xi_at);
  beta ~ normal(beta_mu, beta_sd);
  mu_r ~ normal(0, 1);
  sigma_r ~ exponential(2);
  r_raw ~ std_normal();
  tau_rho ~ exponential(1);
  eps ~ std_normal();
  mu_kappa ~ normal(-2, 1.5);
  sigma_kappa ~ exponential(2);
  kappa_raw ~ std_normal();
  omega_raw ~ normal(1, 1);
  phi_global ~ exponential(1);
  phi_k ~ lognormal(log(phi_global + 1e-8), 0.5);
  tau_phi ~ exponential(2);
  eps2 ~ std_normal();
  delta_phi ~ normal(0, 0.5);
  {
    vector[N_cell_tr] mu = to_vector(alpha)[li_o[1:N_cell_tr]] + to_vector(gamma)[li_d[1:N_cell_tr]]
                           + X[1:N_cell_tr] * beta;
    vector[N] m0 = mu[c0];
    vector[N] mbar = has_l2 .* (omega * mu[c1] + (1 - omega) * mu[c2]) + (1 - has_l2) .* mu[c1];
    vector[N] rho = tanh(r[cl[dyad]] + tau_rho * eps[dyad]);
    vector[N] eta = m0 + is_ar .* rho .* (ly1 - (1 - omega) * mom - mbar) + (1 - is_ar) .* kappa[cl[dyad]]
                    + xi_em[xo] + xi_at[xd];
    vector[N] phi = phi_k[cl[dyad]] .* exp(delta_phi * s + tau_phi * eps2[dyad]);
    target += neg_binomial_2_log_lpmf(y | eta, phi);
    target += -sum(log1m_exp(-phi .* log1p_exp(eta - log(phi))));   // zero truncation: - log P(Y > 0)
  }
}
generated quantities {
  vector[N_t] eta_t;
  vector[N_t] phi_t;
  {
    matrix[N_c, P + 1] af = append_col(alpha, c + tau_em * a_em + theta_em * Z[:, P + 1]);
    matrix[N_c, P + 1] gf = append_col(gamma, tau_at * a_at + theta_at * Z[:, P + 1]);
    vector[N_cell] mu = to_vector(af)[li_o] + to_vector(gf)[li_d] + X * beta;
    // shocks of the training periods, then new ones for T
    vector[N_c * (Q + 1)] xf_em = append_row(xi_em, to_vector(normal_rng(rep_vector(0, N_c), sig_xi_em)));
    vector[N_c * (Q + 1)] xf_at = append_row(xi_at, to_vector(normal_rng(rep_vector(0, N_c), sig_xi_at)));
    for (n in 1:N_t) {
      int d = dyad_t[n];
      int k = cl_t[n];
      real e_rho;
      real e_phi;
      if (d > 0) {
        e_rho = eps[d];
        e_phi = eps2[d];
      } else {                          // dyad unseen in training: dyadic effects drawn from their prior
        e_rho = std_normal_rng();
        e_phi = std_normal_rng();
      }
      real rho = tanh(r[k] + tau_rho * e_rho);
      phi_t[n] = phi_k[k] * exp(delta_phi * s_t[n] + tau_phi * e_phi);
      if (is_ar_t[n] > 0.5) {
        real mbar = has_l2_t[n] > 0.5 ? omega * mu[c1_t[n]] + (1 - omega) * mu[c2_t[n]] : mu[c1_t[n]];
        eta_t[n] = mu[c0_t[n]] + rho * (ly1_t[n] - (1 - omega) * mom_t[n] - mbar);
      } else {
        eta_t[n] = mu[c0_t[n]] + kappa[k];
      }
      eta_t[n] += xf_em[xo_t[n]] + xf_at[xd_t[n]];
    }
  }
}
