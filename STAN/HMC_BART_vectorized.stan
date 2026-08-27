// Volume ARX(2)-ZTNB seul. Le hurdle est externalisé (BART, dbarts).
data {
  int<lower=1> N_pays;

  // Hyper-régression (masse économique)
  int<lower=1> K_Z;
  matrix[N_pays, K_Z] Z_em;
  matrix[N_pays, K_Z] Z_at;

  // Volume ZTNB (train, restriction à N*)
  int<lower=1> N_v;
  int<lower=1> D_v;
  int<lower=1> K_v;
  array[N_v] int<lower=1, upper=D_v> dyad_id_v;
  array[N_v] int<lower=1, upper=N_pays> orig_id_v;
  array[N_v] int<lower=1, upper=N_pays> dest_id_v;
  array[N_v] int<lower=1> flow;
  vector[N_v] log_flow_lag;
  vector[N_v] momentum_v;
  array[N_v] int<lower=0, upper=1> is_emergent_v;
  matrix[N_v, K_v] X_v;
  vector[D_v] log_scale_v;

  // Clusters M49
  int<lower=1> K_clusters;
  array[D_v] int<lower=1, upper=K_clusters> cluster_v;

  // Test OOS (2010 + 2015 empilés)
  int<lower=0> N_test;
  array[N_test] int<lower=0, upper=D_v> dyad_id_test_v;   // 0 = dyade inconnue
  array[N_test] int<lower=1, upper=N_pays> orig_id_test_v;
  array[N_test] int<lower=1, upper=N_pays> dest_id_test_v;
  matrix[N_test, K_v] X_v_test;
  vector[N_test] log_flow_lag_test;
  vector[N_test] momentum_test;
  vector[N_test] is_mig_lag_test;                          // bifurcation OOS
  array[N_test] int<lower=1, upper=K_clusters> cluster_test;

  int<lower=0, upper=1> do_loo;
}

parameters {
  // Effets pays
  real intercept_em;
  vector[K_Z] theta_em;
  real<lower=0> tau_em;
  vector[N_pays] alpha_em_raw;

  real intercept_at;
  vector[K_Z] theta_at;
  real<lower=0> tau_at;
  vector[N_pays] gamma_at_raw;

  // Gravitaire
  vector[K_v] beta_grav;

  // Inertie : hiérarchie global -> M49 -> dyade
  real rho_global_raw;
  real<lower=0> sigma_rho_m49;
  vector[K_clusters] rho_m49_raw;
  real<lower=0> tau_rho;
  vector[D_v] rho_raw;

  // Relais d'émergence (amnésie markovienne)
  real mu_kappa;
  real<lower=0> sigma_kappa;
  vector[K_clusters] kappa_raw;

  // Momentum AR(2) restreint
  real omega_raw;

  // Dispersion
  real<lower=0> phi_disp_global;
  vector<lower=0>[K_clusters] phi_disp_cluster;
  vector[D_v] phi_disp_raw;
  real<lower=0> tau_phi_disp;
  real delta_phi;
}

transformed parameters {
  vector[N_pays] alpha_em = intercept_em + Z_em * theta_em + tau_em * alpha_em_raw;
  vector[N_pays] gamma_at = intercept_at + Z_at * theta_at + tau_at * gamma_at_raw;

  real rho_global = tanh(rho_global_raw);
  vector[K_clusters] rho_m49_lat = rho_global_raw + sigma_rho_m49 * rho_m49_raw;
  vector[K_clusters] rho_m49     = tanh(rho_m49_lat);   // lisible dans (-1,1)
  vector[K_clusters] kappa_m49   = mu_kappa + sigma_kappa * kappa_raw;

  real<lower=0, upper=1> omega = inv_logit(omega_raw);
}

model {
  intercept_em ~ normal(0, 1);
  theta_em     ~ normal(0, 0.5);
  tau_em       ~ normal(0, 0.25);
  alpha_em_raw ~ std_normal();

  intercept_at ~ normal(0, 1);
  theta_at     ~ normal(0, 0.5);
  tau_at       ~ normal(0, 0.25);
  gamma_at_raw ~ std_normal();

  // K_v = 6 : 1 log_D_ij | 2 LB_ij | 3 OL_ij | 4 COL_ij
  //           5 v2x_polyarchy_o_lag5 | 6 intensity_level_o_lag5
  beta_grav[1]     ~ normal(-0.5, 0.5);
  beta_grav[2:4]   ~ normal(0, 1.0);
  beta_grav[5:K_v] ~ normal(0, 1.0);  

  rho_global_raw ~ normal(0, 1);
  sigma_rho_m49  ~ exponential(2);
  rho_m49_raw    ~ std_normal();
  tau_rho        ~ exponential(1);
  rho_raw        ~ std_normal();

  omega_raw ~ normal(1.0, 1);      // centre ~0.82

  mu_kappa    ~ normal(-2.0, 1.5);
  sigma_kappa ~ exponential(2);
  kappa_raw   ~ std_normal();

  phi_disp_global  ~ exponential(1);
  phi_disp_cluster ~ lognormal(log(phi_disp_global + 1e-8), 0.5);
  tau_phi_disp     ~ exponential(2);
  phi_disp_raw     ~ std_normal();
  delta_phi        ~ normal(0, 0.5);

  // ---------- Vraisemblance ZTNB ----------
  {
    vector[D_v] rho_d = tanh(rho_m49_lat[cluster_v] + tau_rho * rho_raw);
    vector[D_v] phi_d = phi_disp_cluster[cluster_v]
                    .* exp(delta_phi * log_scale_v + tau_phi_disp * phi_disp_raw);
    vector[N_v] rho_v = rho_d[dyad_id_v];
    vector[N_v] phi_v = phi_d[dyad_id_v];

    vector[N_v] mu_dt = alpha_em[orig_id_v] + gamma_at[dest_id_v] + X_v * beta_grav;

    vector[N_v] ar_pred;
    for (n in 1:N_v) {
      if (is_emergent_v[n] == 1) {
        ar_pred[n] = mu_dt[n] + kappa_m49[cluster_v[dyad_id_v[n]]];
      } else {
        real L_bar = log_flow_lag[n] - (1 - omega) * momentum_v[n];
        ar_pred[n] = mu_dt[n] + rho_v[n] * (L_bar - mu_dt[n]);
      }
    }

    target += neg_binomial_2_log_lpmf(flow | ar_pred, phi_v);
    vector[N_v] log_p0 = -phi_v .* log1p_exp(ar_pred - log(phi_v));
    target += -sum(log1m_exp(log_p0));
  }
}

generated quantities {
  vector[N_test] mu_dt_test;
  vector[N_test] phi_test;
  vector[do_loo ? N_v : 0] log_lik_v;

  {
    vector[D_v] rho_d = tanh(rho_m49_lat[cluster_v] + tau_rho * rho_raw);
    vector[D_v] phi_d = phi_disp_cluster[cluster_v]
                    .* exp(delta_phi * log_scale_v + tau_phi_disp * phi_disp_raw);

    vector[N_test] mu_full = alpha_em[orig_id_test_v] + gamma_at[dest_id_test_v]
                             + X_v_test * beta_grav;

    for (n in 1:N_test) {
      int d_v = dyad_id_test_v[n];
      int k   = cluster_test[n];
      if (is_mig_lag_test[n] < 0.5) {
        mu_dt_test[n] = mu_full[n] + kappa_m49[k];
        phi_test[n]   = (d_v > 0) ? phi_d[d_v] : phi_disp_cluster[k];
      } else {
        real L_bar = log_flow_lag_test[n] - (1 - omega) * momentum_test[n];
        if (d_v > 0) {
          mu_dt_test[n] = mu_full[n] + rho_d[d_v] * (L_bar - mu_full[n]);
          phi_test[n]   = phi_d[d_v];
        } else {
          mu_dt_test[n] = mu_full[n] + rho_global * (L_bar - mu_full[n]);
          phi_test[n]   = phi_disp_cluster[k];
        }
      }
    }

    if (do_loo) {
      vector[N_v] mu_dt_tr = alpha_em[orig_id_v] + gamma_at[dest_id_v] + X_v * beta_grav;
      for (n in 1:N_v) {
        int d = dyad_id_v[n];
        real L_bar = log_flow_lag[n] - (1 - omega) * momentum_v[n];
        real ar_n  = mu_dt_tr[n] + rho_d[d] * (L_bar - mu_dt_tr[n]);
        real lp0   = -phi_d[d] * log1p_exp(ar_n - log(phi_d[d]));
        log_lik_v[n] = neg_binomial_2_log_lpmf(flow[n] | ar_n, phi_d[d]) - log1m_exp(lp0);
      }
    }
  }
}
