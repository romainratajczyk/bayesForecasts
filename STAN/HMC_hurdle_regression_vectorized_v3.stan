// ============================================================================
// HMC_hurdle_regression_vectorized_v3.stan
// Corrections de performance vs v2 :
//   (1) SUPPRESSION de alpha_raw (vector[D_h] sans prior ni usage -> ~2 862
//       dimensions plates qui forcent NUTS vers max_treedepth a chaque
//       iteration ; c'est le principal responsable des ~6 s/iteration).
//   (2) p_hurdle sorti de la boucle test et declare UNE FOIS au niveau
//       superieur des generated quantities (en v2 il etait recalcule
//       N_test x N_h fois par draw ET jamais ecrit car local au bloc for).
//       Restreint aux lignes de calibration via calib_idx (2010 + 2005).
//   (3) logit_p, lag_effect, mu_dt, ar_pred, rho_d, phi_disp_d demotes de
//       transformed parameters vers des locals du bloc model : plus rien de
//       dimension N_h / N_v / D_v n'est ecrit dans les CSV (les transformed
//       parameters sont TOUS sauvegardes a chaque draw -> ~55k colonnes en v2).
//   (4) Vraisemblance hurdle via bernoulli_logit_glm_lpmf (gradients
//       analytiques, 2-4x plus rapide que X_h * beta_h + bernoulli_logit).
//   (5) Troncature ZTNB reecrite avec log1p_exp (elementwise, stable) :
//       log P(Y=0) = -phi .* log1p_exp(ar_pred - log(phi)).
//   (6) GQ test vectorise (matrice-vecteur une fois par draw au lieu de
//       N_test dot_product).
//   (7) ICAR : ajout d'une contrainte douce sum-to-zero (le niveau de u_em
//       etait confondu avec intercept_h_em -> ridge, pas de vraie ancre).
//
// ============================================================================

data {
  int<lower=1> N_pays;

  // Hyper-regression
  int<lower=1> K_Z;
  matrix[N_pays, K_Z] Z_em;
  matrix[N_pays, K_Z] Z_at;

  // Hurdle (train)
  int<lower=1> N_h;
  int<lower=1> D_h;
  int<lower=1> K_h;                       // SANS is_mig_lag
  array[N_h] int<lower=1, upper=D_h> dyad_id_h;
  array[N_h] int<lower=1, upper=N_pays> orig_id_h;
  array[N_h] int<lower=1, upper=N_pays> dest_id_h;
  array[N_h] int<lower=0, upper=1> is_mig; // coeff qui varie par cluster M49 et partiellement poolé
  vector[N_h] is_mig_lag;
  matrix[N_h, K_h] X_h;

  // Volume ZTNB (restriction a N*)
  int<lower=1> N_v;
  int<lower=1> D_v;
  int<lower=1> K_v;
  array[N_v] int<lower=1, upper=D_v> dyad_id_v;
  array[N_v] int<lower=1, upper=N_pays> orig_id_v;
  array[N_v] int<lower=1, upper=N_pays> dest_id_v;
  array[N_v] int<lower=1> flow;
  vector[N_v] log_flow_lag;
  array[N_v] int<lower=0, upper=1> is_emergent_v;
  matrix[N_v, K_v] X_v;

  // Clusters M49
  int<lower=1> K_clusters;
  array[D_h] int<lower=1, upper=K_clusters> cluster_h;
  array[D_v] int<lower=1, upper=K_clusters> cluster_v;

  // Calibration du seuil : lignes hurdle dont p_hurdle doit etre exporte
  int<lower=0> N_calib;
  array[N_calib] int<lower=1, upper=N_h> calib_idx;

  // Test OOS
  int<lower=0> N_test;
  array[N_test] int<lower=1, upper=D_h> dyad_id_test_h;   // conserve (compat)
  array[N_test] int<lower=0, upper=D_v> dyad_id_test_v;   // 0 = dyade inconnue
  array[N_test] int<lower=1, upper=N_pays> orig_id_test_v;
  array[N_test] int<lower=1, upper=N_pays> dest_id_test_v;
  matrix[N_test, K_h] X_h_test;
  vector[N_test] is_mig_lag_test;
  matrix[N_test, K_v] X_v_test;
  vector[N_test] log_flow_lag_test;
  array[N_test] int<lower=1, upper=K_clusters> cluster_test_h;

  // Flags
  int<lower=0, upper=1> do_ppc;
  int<lower=0, upper=1> do_loo;

  // Graphe de contiguite (prior spatial ICAR sur l'emission hurdle)
  int<lower=0> N_edges;
  array[N_edges] int<lower=1, upper=N_pays> node1;
  array[N_edges] int<lower=1, upper=N_pays> node2;
}

transformed data {
  // Pre-resolution du double index cluster_h[dyad_id_h] (une fois pour toutes)
  array[N_h] int cluster_obs_h;
  for (n in 1:N_h)
    cluster_obs_h[n] = cluster_h[dyad_id_h[n]];
}

parameters {
  // A. Hurdle
  vector[K_h] beta_h;
  real mu_beta_lag; 
  real<lower=0> sigma_beta_lag;
  vector[K_clusters] beta_lag_raw;
  // NOTE : alpha_raw (vector[D_h]) supprime. S'il faut un effet aleatoire
  // dyadique plus tard, le reintroduire AVEC prior + non-centrage.

  real intercept_h_em;
  vector[K_Z] theta_h_em;
  real<lower=0> tau_h_em;
  vector[N_pays] alpha_h_em_raw;

  real intercept_h_at;
  vector[K_Z] theta_h_at;
  real<lower=0> tau_h_at;
  vector[N_pays] gamma_h_at_raw;

  // Champ spatial (emission hurdle), non-centre : echelle portee par tau_u_em
  vector[N_pays] u_em;
  real<lower=0> tau_u_em;

  // B. Volume ARX
  real intercept_em;
  vector[K_Z] theta_em;
  real<lower=0> tau_em;
  vector[N_pays] alpha_em_raw;

  real intercept_at;
  vector[K_Z] theta_at;
  real<lower=0> tau_at;
  vector[N_pays] gamma_at_raw;

  vector[K_v] beta_grav;
  real rho_global_raw;
  real<lower=0> sigma_rho_m49;        // dispersion inter-cluster 
  vector[K_clusters] rho_m49_raw;     // non-centre (nouveau)
  real<lower=0> tau_rho;              // dispersion intra-cluster 
  vector[D_v] rho_raw;
  // Coût  hiérarchique amnésie markovienne et log_flow_lag = -infty
  real mu_kappa;
  real<lower=0> sigma_kappa;
  vector[K_clusters] kappa_raw;

  // C. Dispersion
  real<lower=0> phi_disp_global;
  vector<lower=0>[K_clusters] phi_disp_cluster;
  vector[D_v] phi_disp_raw;
  real<lower=0> tau_phi_disp;
}

transformed parameters {
  // UNIQUEMENT des quantites de petite dimension (N_pays, K_clusters) :
  // necessaires au bloc model ET aux GQ, cout disque negligeable.
  vector[K_clusters] beta_lag_m49 = mu_beta_lag + sigma_beta_lag * beta_lag_raw;

  vector[N_pays] alpha_h_em = intercept_h_em + Z_em * theta_h_em
                              + tau_h_em * alpha_h_em_raw
                              + tau_u_em * u_em;
  vector[N_pays] gamma_h_at = intercept_h_at + Z_at * theta_h_at
                              + tau_h_at * gamma_h_at_raw;

  vector[N_pays] alpha_em = intercept_em + Z_em * theta_em + tau_em * alpha_em_raw;
  vector[N_pays] gamma_at = intercept_at + Z_at * theta_at + tau_at * gamma_at_raw;

  real rho_global = tanh(rho_global_raw);

  // Niveau cluster M49 (echelle latente, non-centre)
  vector[K_clusters] rho_m49_lat = rho_global_raw + sigma_rho_m49 * rho_m49_raw;
  // Version interpretable, dans (-1, 1) : c'est le rho du cluster quand
  // la deviation dyadique est nulle
  vector[K_clusters] rho_m49 = tanh(rho_m49_lat); // donne malgré tout la quantité lisible dans (−1,1) pour tableaux et violons ; on ne compose pas deux tanh à la suite
  //médiane des rho _d du cluster plutôt que leur moyenne (bien)
  vector[K_clusters] kappa_m49 = mu_kappa + sigma_kappa * kappa_raw;
}

model {
  // ---------- A. Priors Hurdle ----------
  intercept_h_em ~ normal(-1.0, 1.5);
  theta_h_em ~ normal(0, 0.5);
  tau_h_em ~ normal(0, 0.25);
  alpha_h_em_raw ~ std_normal();

  intercept_h_at ~ normal(0, 1.0);
  theta_h_at ~ normal(0, 0.5);
  tau_h_at ~ normal(0, 0.25);
  gamma_h_at_raw ~ std_normal();

  mu_beta_lag ~ normal(2.0, 2.5);
  sigma_beta_lag ~ exponential(1);
  beta_lag_raw ~ std_normal();

  // Ordre de X_h : cf. en-tete. K_h = 12 (sans is_mig_lag).
  beta_h[1] ~ normal(-0.5, 0.5);          // log_D_ij
  beta_h[2] ~ normal(-0.5, 0.5);          // log_D_ij_sq
  beta_h[3] ~ normal(0, 2);               // COL_ij
  beta_h[4] ~ normal(0, 2);               // OL_ij
  beta_h[5:(K_h - 1)] ~ normal(0, 1.0);   // geopolitiques + A2_log
  beta_h[K_h] ~ normal(1.0, 1.0);         // logit_rf (prior positif)

  // ---------- Prior spatial ICAR (emission hurdle) ----------
  target += -0.5 * dot_self(u_em[node1] - u_em[node2]);
  // Ancrage du niveau : l'ICAR est invariant par translation ; sans cette
  // contrainte, le niveau de u_em est confondu avec intercept_h_em (ridge).
  sum(u_em) ~ normal(0, 0.001 * N_pays);
  // Prior propre pour les composantes deconnectees / iles
  // (Freni-Sterrantino et al.) — garde le champ identifie hors du graphe.
  u_em ~ normal(0, 1);
  tau_u_em ~ normal(0, 0.5);              // half-normal via <lower=0>

  // ---------- B. Priors Volume ----------
  intercept_em ~ normal(0, 1);
  theta_em ~ normal(0, 0.5);
  tau_em ~ normal(0, 0.25);
  alpha_em_raw ~ std_normal();

  intercept_at ~ normal(0, 1);
  theta_at ~ normal(0, 0.5);
  tau_at ~ normal(0, 0.25);
  gamma_at_raw ~ std_normal();

  beta_grav[1] ~ normal(-0.5, 0.5);       // log_D_ij
  beta_grav[2] ~ normal(-0.5, 0.5);       // log_D_ij_sq
  beta_grav[3:5] ~ normal(0, 2.0);        // LB_ij, OL_ij, COL_ij
  beta_grav[6:7] ~ normal(0, 1.0);        // t_2000, t_2000_sq
  beta_grav[8:K_v] ~ normal(0, 1.0);      // geopolitiques standardisees

  rho_global_raw ~ normal(0.5, 0.5);
  sigma_rho_m49 ~ exponential(2);     // shrinkage des M49 vers l'ancre globale
  rho_m49_raw ~ std_normal();
  tau_rho ~ exponential(2);
  rho_raw ~ std_normal();

  // Priors d'émergence
  mu_kappa ~ normal(-2.0, 1.5);
  sigma_kappa ~ exponential(2);
  kappa_raw ~ std_normal();

  // ---------- C. Priors Dispersion ----------
  phi_disp_global ~ exponential(1);
  phi_disp_cluster ~ lognormal(log(phi_disp_global + 1e-8), 0.5);
  tau_phi_disp ~ exponential(2);
  phi_disp_raw ~ std_normal();

  // ---------- Vraisemblance Hurdle (GLM primitive) ----------
  {
    vector[N_h] eta0 = alpha_h_em[orig_id_h] + gamma_h_at[dest_id_h]
                       + beta_lag_m49[cluster_obs_h] .* is_mig_lag;
    target += bernoulli_logit_glm_lpmf(is_mig | X_h, eta0, beta_h);
  }

  // ---------- Vraisemblance Volume ZTNB (locals : rien n'est ecrit) ----------
  {
    vector[D_v] rho_d = tanh(rho_m49_lat[cluster_v] + tau_rho * rho_raw);
    vector[D_v] phi_d = phi_disp_cluster[cluster_v]
                        .* exp(tau_phi_disp * phi_disp_raw);
    vector[N_v] rho_v = rho_d[dyad_id_v];
    vector[N_v] phi_v = phi_d[dyad_id_v];

    vector[N_v] mu_dt   = alpha_em[orig_id_v] + gamma_at[dest_id_v]
                          + X_v * beta_grav;
    // Bifurcation structurelle de l'espérance
    vector[N_v] ar_pred;
    for (n in 1:N_v) {
      if (is_emergent_v[n] == 1) {
        ar_pred[n] = mu_dt[n] + kappa_m49[cluster_v[dyad_id_v[n]]];
      } else {
        ar_pred[n] = mu_dt[n] + rho_v[n] * (log_flow_lag[n] - mu_dt[n]);
      }
  }
    //vector[N_v] ar_pred = mu_dt + rho_v .* (log_flow_lag - mu_dt);

    // NB2 non tronquee
    target += neg_binomial_2_log_lpmf(flow | ar_pred, phi_v);

    // Penalite de troncature : log P(Y=0) = phi*(log phi - log(phi + mu))
    //                                    = -phi * log1p(mu/phi)
    //                                    = -phi * log1p_exp(ar_pred - log phi)
    vector[N_v] log_p0 = -phi_v .* log1p_exp(ar_pred - log(phi_v));
    target += -sum(log1m_exp(log_p0));
  }
}

generated quantities {
  // --- Sorties principales (toutes de dimension raisonnable) ---
  vector[N_calib] p_hurdle;                 // seulement les annees de calibration
  vector[N_test] prob_mig_test;
  vector[N_test] mu_dt_test;
  vector[N_test] phi_test;
     
  real rho_global_monitor = rho_global;

  vector[do_loo ? N_h : 0] log_lik_h;
  vector[do_loo ? N_v : 0] log_lik_v;
  array[do_ppc ? N_h : 0] int is_mig_hat;

  {
    // Recomputations locales, UNE fois par draw sauvegarde, sans autodiff.
    vector[D_v] rho_d = tanh(rho_m49_lat[cluster_v] + tau_rho * rho_raw);
    vector[D_v] phi_d = phi_disp_cluster[cluster_v]
                        .* exp(tau_phi_disp * phi_disp_raw);

    // Hurdle train
    vector[N_h] logit_p = alpha_h_em[orig_id_h] + gamma_h_at[dest_id_h]
                          + X_h * beta_h
                          + beta_lag_m49[cluster_obs_h] .* is_mig_lag;
    p_hurdle = inv_logit(logit_p[calib_idx]);

    // Hurdle test — vectorise, lag inclus (coherence train/test)
    vector[N_test] logit_p_test = alpha_h_em[orig_id_test_v]
                                  + gamma_h_at[dest_id_test_v]
                                  + X_h_test * beta_h
                                  + beta_lag_m49[cluster_test_h] .* is_mig_lag_test;
    prob_mig_test = inv_logit(logit_p_test);

    // // Volume test — matrice-vecteur une fois, puis boucle legere pour le if
    // vector[N_test] mu_full = alpha_em[orig_id_test_v] + gamma_at[dest_id_test_v]
    //                          + X_v_test * beta_grav;
    // for (n in 1:N_test) {
    //   int d_v = dyad_id_test_v[n];
    //   if (d_v > 0) {
    //     mu_dt_test[n] = mu_full[n] + rho_d[d_v] * (log_flow_lag_test[n] - mu_full[n]);
        
    //     phi_test[n]   = phi_d[d_v];
    //   } else {
    //     // Dyade jamais vue en train : inertie globale + phi du cluster
    //     mu_dt_test[n] = mu_full[n] + tanh(rho_m49_lat[cluster_test_h[n]]) * (log_flow_lag_test[n] - mu_full[n]);
    //     phi_test[n]   = phi_disp_cluster[cluster_test_h[n]];
    //   }
    // }
    // Volume test matrice-vecteur une fois, puis boucle legere pour le if
  vector[N_test] mu_full = alpha_em[orig_id_test_v] + gamma_at[dest_id_test_v] + X_v_test * beta_grav;

  for (n in 1:N_test) {
    int d_v = dyad_id_test_v[n];
    int k = cluster_test_h[n];
  
  // Bifurcation OOS : is_mig_lag_test == 0 sature les Faux Positifs, les trous et les émergences pures.
    // Remplace : if (is_mig_lag_test[n] == 0) {
    if (is_mig_lag_test[n] < 0.5) {
      mu_dt_test[n] = mu_full[n] + kappa_m49[k];
      phi_test[n] = (d_v > 0) ? phi_d[d_v] : phi_disp_cluster[k];
    } else {
    // Application de la chaîne ARX(1) pour les continuités strictes
      if (d_v > 0) {
        mu_dt_test[n] = mu_full[n] + rho_d[d_v] * (log_flow_lag_test[n] - mu_full[n]);
        phi_test[n] = phi_d[d_v];
      } else {
        mu_dt_test[n] = mu_full[n] + rho_global * (log_flow_lag_test[n] - mu_full[n]);
        phi_test[n] = phi_disp_cluster[k];
      }
    }
  }

    
    

    if (do_loo) {
      for (n in 1:N_h)
        log_lik_h[n] = bernoulli_logit_lpmf(is_mig[n] | logit_p[n]);
      vector[N_v] mu_dt_tr = alpha_em[orig_id_v] + gamma_at[dest_id_v]
                             + X_v * beta_grav;
      for (n in 1:N_v) {
        int d = dyad_id_v[n];
        real ar_n = mu_dt_tr[n] + rho_d[d] * (log_flow_lag[n] - mu_dt_tr[n]);
        real lp0  = -phi_d[d] * log1p_exp(ar_n - log(phi_d[d]));
        log_lik_v[n] = neg_binomial_2_log_lpmf(flow[n] | ar_n, phi_d[d])
                       - log1m_exp(lp0);
      }
    }

    if (do_ppc) {
      for (n in 1:N_h)
        is_mig_hat[n] = bernoulli_logit_rng(logit_p[n]);
    }
  }
}
