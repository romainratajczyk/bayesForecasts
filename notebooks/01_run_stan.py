# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import cmdstanpy
from cmdstanpy import CmdStanModel

cmdstan_base_dir = "/home/onyxia/work/cmdstan"

# Détection automatique du sous-dossier CmdStan
try:
    installed_versions = [d for d in os.listdir(cmdstan_base_dir) if d.startswith("cmdstan-")]
    cmdstan_path = os.path.join(cmdstan_base_dir, installed_versions[0])
    cmdstanpy.set_cmdstan_path(cmdstan_path)
    print(f"Liaison CmdStan établie sur : {cmdstan_path}")
except IndexError:
    raise FileNotFoundError("Les binaires CmdStan sont introuvables. Exécutez cmdstanpy.install_cmdstan(dir=cmdstan_base_dir) au préalable.")

# %%
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve , roc_auc_score
from itertools import product

#from sklearn.ensemble import RandomForestClassifier
from scipy.special import logit as scipy_logit

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_PATH  = "../data/panel_june_ready.csv"
STAN_FILE  = "../STAN/HMC_ARX_ZTNB.stan"
OUTPUT_DIR = "./stan_outputs_tmux"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% [markdown]
# ### tester gdpcap_lag1 pour l'horizon 1 
# ### ajouter LA LL et urban pour PSE et DOM-TOM

# %%
# Sampling parameters
N_CHAINS        = 4
PARALLEL_CHAINS = 4
ITER_WARMUP     = 1000
ITER_SAMPLING   = 1300
THIN            = 2
MAX_TREEDEPTH   = 12
ADAPT_DELTA     = 0.95
N_DRAWS         = ITER_SAMPLING // THIN

# %%
df_main = pd.read_csv(DATA_PATH)
df = df_main[df_main['orig'] != df_main['dest']].copy()

PAYS_EXCLURE = {
    'SSD', 'MNE', 'TLS', 'CUW',
    'GUM', 'MYT', 'VIR', 'CLI', # on va essayer d'intégrer ceux-là manuellement (manque le PIB)
}
df = df[
    ~df['orig'].isin(PAYS_EXCLURE) &
    ~df['dest'].isin(PAYS_EXCLURE)
].copy()

df = df.sort_values(['orig', 'dest', 'year']).reset_index(drop=True)
print(f"{df['orig'].nunique()} pays après exclusions")

# %%
# clustering M49

ISO3_TO_M49_SUBREGION = {
    'DNK': 11, 'EST': 11, 'FIN': 11, 'ISL': 11, 'IRL': 11, 'LVA': 11, 'LTU': 11, 'NOR': 11, 'SWE': 11, 'GBR': 11,
    'ALB': 12, 'AND': 12, 'BIH': 12, 'HRV': 12, 'GRC': 12, 'ITA': 12, 'MLT': 12, 'MNE': 12, 'MKD': 12, 'PRT': 12, 'SRB': 12, 'SVN': 12, 'ESP': 12,
    'AUT': 13, 'BEL': 13, 'FRA': 13, 'DEU': 13, 'LIE': 13, 'LUX': 13, 'MCO': 13, 'NLD': 13, 'CHE': 13,
    'BLR': 14, 'BGR': 14, 'CZE': 14, 'HUN': 14, 'POL': 14, 'MDA': 14, 'ROU': 14, 'RUS': 14, 'SVK': 14, 'UKR': 14,
    'DZA': 15, 'EGY': 15, 'LBY': 15, 'MAR': 15, 'SDN': 15, 'TUN': 15, 'ESH': 15,
    'BEN': 16, 'BFA': 16, 'CPV': 16, 'CIV': 16, 'GMB': 16, 'GHA': 16, 'GIN': 16, 'GNB': 16, 'LBR': 16, 'MLI': 16, 'MRT': 16, 'NER': 16, 'NGA': 16, 'SEN': 16, 'SLE': 16, 'TGO': 16,
    'BDI': 17, 'COM': 17, 'DJI': 17, 'ERI': 17, 'ETH': 17, 'KEN': 17, 'MDG': 17, 'MWI': 17, 'MUS': 17, 'MOZ': 17, 'REU': 17, 'RWA': 17, 'SYC': 17, 'SOM': 17, 'SSD': 17, 'TZA': 17, 'UGA': 17, 'ZMB': 17, 'ZWE': 17,
    'AGO': 18, 'CMR': 18, 'CAF': 18, 'TCD': 18, 'COD': 18, 'COG': 18, 'GNQ': 18, 'GAB': 18, 'STP': 18,
    'BWA': 19, 'LSO': 19, 'NAM': 19, 'ZAF': 19, 'SWZ': 19,
    'CAN': 21, 'MEX': 21, 'USA': 21,
    'BLZ': 22, 'CRI': 22, 'SLV': 22, 'GTM': 22, 'HND': 22, 'NIC': 22, 'PAN': 22,
    'ATG': 23, 'BHS': 23, 'BRB': 23, 'CUB': 23, 'DMA': 23, 'DOM': 23, 'GLP': 23, 'GRD': 23, 'HTI': 23, 'JAM': 23, 'KNA': 23, 'LCA': 23, 'MTQ': 23, 'VCT': 23, 'TTO': 23, 'ABW': 23, 'PRI': 23,
    'ARG': 24, 'BOL': 24, 'BRA': 24, 'CHL': 24, 'COL': 24, 'ECU': 24, 'GUF': 24, 'GUY': 24, 'PRY': 24, 'PER': 24, 'SUR': 24, 'URY': 24, 'VEN': 24,
    'CHN': 30, 'HKG': 30, 'JPN': 30, 'KOR': 30, 'MAC': 30, 'MNG': 30, 'PRK': 30,
    'AFG': 34, 'BGD': 34, 'BTN': 34, 'IND': 34, 'IRN': 34, 'MDV': 34, 'NPL': 34, 'PAK': 34, 'LKA': 34,
    'BRN': 35, 'KHM': 35, 'IDN': 35, 'LAO': 35, 'MYS': 35, 'MMR': 35, 'PHL': 35, 'SGP': 35, 'THA': 35, 'TLS': 35, 'VNM': 35,
    'ARM': 145, 'AZE': 145, 'BHR': 145, 'CYP': 145, 'GEO': 145, 'IRQ': 145, 'ISR': 145, 'JOR': 145, 'KWT': 145, 'LBN': 145, 'OMN': 145, 'QAT': 145, 'SAU': 145, 'PSE': 145, 'SYR': 145, 'TUR': 145, 'ARE': 145, 'YEM': 145,
    'KAZ': 143, 'KGZ': 143, 'TJK': 143, 'TKM': 143, 'UZB': 143,
    'AUS': 53, 'FJI': 53, 'NZL': 53, 'PNG': 53, 'SLB': 53, 'VUT': 53, 'WSM': 53, 'TON': 53, 'KIR': 53, 'FSM': 53, 'GUM': 53, 'NCL': 53, 'PYF': 53,
}

SUBREGION_LABELS = {
    11: 'Europe du Nord', 12: 'Europe du Sud', 13: "Europe de l'Ouest", 14: "Europe de l'Est",
    15: 'Afrique du Nord', 16: "Afrique de l'Ouest", 17: "Afrique de l'Est", 18: 'Afrique Centrale', 19: 'Afrique Australe',
    21: 'Amerique du Nord', 22: 'Amerique Centrale', 23: 'Caraibes', 24: 'Amerique du Sud',
    30: "Asie de l'Est", 34: 'Asie du Sud', 35: 'Asie du Sud-Est',
    143: 'Asie Centrale', 145: "Asie de l'Ouest", 53: 'Oceanie', 99: 'Non classifie'
}

df['m49_brut'] = df['orig'].map(lambda x: ISO3_TO_M49_SUBREGION.get(str(x).upper(), 99))
_UNIQUE_M49_PRESENT = sorted(df['m49_brut'].unique())
_M49_TO_STAN = {m49: i + 1 for i, m49 in enumerate(_UNIQUE_M49_PRESENT)}
stan_to_m49 = {v: k for k, v in _M49_TO_STAN.items()}
df['continent_orig'] = df['m49_brut'].map(_M49_TO_STAN)
K_clusters = len(_M49_TO_STAN)
print(f"{K_clusters} clusters M49")

# %%
# df['is_migration'] = (df['flow'] > 0).astype(int)
# df['log_flow']     = np.where(df['flow'] > 0, np.log(df['flow']), np.nan)
# df['log_flow_lag'] = df.groupby(['orig', 'dest'])['log_flow'].shift(1)
# df['is_mig_lag']   = df.groupby(['orig', 'dest'])['is_migration'].shift(1)
# df['log_D_ij']     = np.log(df['D_ij'].replace(0, np.nan))
# df['log_D_ij_sq']  = df['log_D_ij'] ** 2
# df['logD_times_LB'] = df['log_D_ij'] * df['LB_ij']


df['dyad']          = df['orig'] + "_" + df['dest']

# df['instability_o'] = df['v2x_clphy_o_lag1'] - df['v2x_polyarchy_o_lag1']
# df['instability_d'] = df['v2x_clphy_d_lag1'] - df['v2x_polyarchy_d_lag1']

df = df.sort_values(['orig', 'dest', 'year']).reset_index(drop=True)

# inutile déjà calculé par build_dataset, à supprimer: 
# for col_base, group_key in [
#     ('intensity_level_o_lag1', 'orig'),
#     ('type_of_conflict_o_lag1', 'orig'),
#     ('intensity_level_d_lag1', 'dest'),
#     ('type_of_conflict_d_lag1', 'dest'),
# ]:
#     new_col = f'{col_base}_persist'
#     df[new_col] = df.groupby([group_key, 'dest' if group_key == 'orig' else 'orig'])[col_base] \
#         .transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean())

df = df.dropna(subset=['is_mig_lag']).reset_index(drop=True)

GRAVITY_VARS_RAW = ['P_it', 'P_jt', 'PSR_i', 'PSR_j', 'IMR_it', 'IMR_jt', 'urban_it', 'urban_jt', 'LA_i', 'LA_j']
for raw in GRAVITY_VARS_RAW:
    df[f'log_{raw}'] = np.log(df[raw].replace(0, np.nan)) # créer les variables log

# %%

# HURDLE_VARS RF avec colinéarité (de la soutenance)
# HURDLE_VARS = [
#     'log_D_ij', 'log_D_ij_sq', 'COL_ij', 'OL_ij',
#     'v2x_polyarchy_o_lag1', 'v2x_clphy_o_lag1', 'intensity_level_o_lag1',
#     'v2x_polyarchy_d_lag1', 'v2x_clphy_d_lag1', 'intensity_level_d_lag1',
# ]

# HURDLE_VARS sain avec XGBoost 

# à mettre après HURDLE_VARS = ['logit_xgb']
HURDLE_VARS_xgb = [
    'log_D_ij', 'log_D_ij_sq', 'COL_ij', 'OL_ij',
    'v2x_polyarchy_o_lag1', 'v2x_clphy_o_lag1', 'intensity_level_o_lag1',
    'v2x_polyarchy_d_lag1', 'v2x_clphy_d_lag1', 'intensity_level_d_lag1',
] # ne sont pas des covariables de la régression, simplement injectées dans le XGB
HURDLE_VARS = [] # covariables de la régression (aucune, on ajoutera logit_xgb et is_mig_lag)


X_VOL_COLS = [
    'log_D_ij', 'LB_ij', 'OL_ij', 'COL_ij', 't_2000', 't_2000_sq',
    'v2x_polyarchy_o_lag1', 'v2x_clphy_o_lag1', 'intensity_level_o_lag1',
    'v2x_polyarchy_d_lag1', 'v2x_clphy_d_lag1', 'intensity_level_d_lag1', 'type_of_conflict_d_lag1',
]

K_grav = len(X_VOL_COLS)
K_h = len(HURDLE_VARS)

df_train = df[df['year'] <= 2010].copy()
df_test = df[df['year'] == 2015].copy()
df = df_train

# %%
HURDLE_REQUIRED = HURDLE_VARS + [ 'is_migration', 'dyad', 'continent_orig',
                                 #'is_mig_lag',
                                 ] 
# covariables + is_mig_lag ne devant pas être standardisée et occupant une place théorique particulière (hystérésis) 
# + variables structurelles  dont Stan a besoin pour l'entraînement et la vraisemblance 
# (dyad pour les effets fixes alpha_i et gamma_j, continent_orig pour les effets de cluster M49)
df_hurdle = df.dropna(subset=HURDLE_REQUIRED).copy().reset_index(drop=True)

VOLUME_REQUIRED = X_VOL_COLS + ['flow', 'log_flow_lag', 'dyad', 'continent_orig']
df_volume = df[df['flow'] > 0].dropna(subset=VOLUME_REQUIRED).copy().reset_index(drop=True)

N_h, N_v = len(df_hurdle), len(df_volume)
print(f"Hurdle : {N_h:,} obs | Volume : {N_v:,} obs")

# %%
out_degree = df_hurdle.groupby(['orig', 'year'])['is_mig_lag'].sum().reset_index(name='out_degree_o')
in_degree  = df_hurdle.groupby(['dest', 'year'])['is_mig_lag'].sum().reset_index(name='in_degree_d')
df_hurdle  = df_hurdle.merge(out_degree, on=['orig', 'year'], how='left')
df_hurdle  = df_hurdle.merge(in_degree,  on=['dest', 'year'], how='left')
df_hurdle['transitivity_proxy'] = (
    df_hurdle['out_degree_o'].fillna(0) * df_hurdle['in_degree_d'].fillna(0)
)

out_deg_agg = df_hurdle.groupby('orig')['out_degree_o'].mean().reset_index()
in_deg_agg  = df_hurdle.groupby('dest')['in_degree_d'].mean().reset_index()
df_test = df_test.merge(out_deg_agg, on='orig', how='left')
df_test = df_test.merge(in_deg_agg,  on='dest', how='left')
df_test['transitivity_proxy'] = (
    df_test['out_degree_o'].fillna(0) * df_test['in_degree_d'].fillna(0)
)

# %%
# entrainement du XGB (hurdle) 
XGB_VARS = HURDLE_VARS_xgb + [
    'log_gdpcap_o_lag1', 'log_gdpcap_d_lag1', 'log_gdpcap_diff',
    'log_P_it', 'log_P_jt',
    'is_mig_lag',
    'PSR_i', 'PSR_j',
    'IMR_it', 'IMR_jt',
    'urban_it', 'urban_jt',
    'LL_i', 'LL_j',
    'LA_i', 'LA_j',
    'LB_ij', 'logD_times_LB',
    'type_of_conflict_o_lag1', 'type_of_conflict_d_lag1',
    'transitivity_proxy',
    'v2x_polyarchy_o_lag5', 'v2x_clphy_o_lag5', 'intensity_level_o_lag5', 'type_of_conflict_o_lag5',
    'v2x_polyarchy_d_lag5', 'v2x_clphy_d_lag5', 'intensity_level_d_lag5', 'type_of_conflict_d_lag5',
    'log_stock_lag',
    'any_conflict_o_window', 'max_conflict_o_window', 'any_intense_o_window', 'any_intl_o_window',
    'any_conflict_d_window', 'max_conflict_d_window', 'any_intense_d_window', 'any_intl_d_window',
    'new_conflict_o', 'new_conflict_d', 'persistent_conflict_o', 'persistent_conflict_d',
] + ['is_mig_lag'] 
# is_mig_lag uniquement dans le XGBoost (avantage: qualité prédictive OOS et interactions non linéaires, mais pas d'interprétation) 


# XGB_VARS = [
#     'log_D_ij', 'log_D_ij_sq', 'COL_ij', 'OL_ij',
#     'v2x_polyarchy_o_lag1', 'v2x_clphy_o_lag1', 'intensity_level_o_lag1',
#     'v2x_polyarchy_d_lag1', 'v2x_clphy_d_lag1', 'intensity_level_d_lag1',
#     'log_gdpcap_o_lag1', 'log_gdpcap_d_lag1', 'log_gdpcap_diff',
#     'log_P_it', 'log_P_jt',
#     'is_mig_lag',
#     'PSR_i', 'PSR_j',
#     'IMR_it', 'IMR_jt',
#     'urban_it', 'urban_jt',
#     'LL_i', 'LL_j',
#     'LA_i', 'LA_j',
#     'LB_ij', 'logD_times_LB',
#     'type_of_conflict_o_lag1', 'type_of_conflict_d_lag1',
#     'transitivity_proxy',
#     'v2x_polyarchy_o_lag5', 'v2x_clphy_o_lag5', 'intensity_level_o_lag5', 'type_of_conflict_o_lag5',
#     'v2x_polyarchy_d_lag5', 'v2x_clphy_d_lag5', 'intensity_level_d_lag5', 'type_of_conflict_d_lag5',
#     'log_stock_lag',
#     'any_conflict_o_window', 'max_conflict_o_window', 'any_intense_o_window', 'any_intl_o_window',
#     'any_conflict_d_window', 'max_conflict_d_window', 'any_intense_d_window', 'any_intl_d_window',
#     'new_conflict_o', 'new_conflict_d', 'persistent_conflict_o', 'persistent_conflict_d',
# ]
eps = 10e-6 

XGB_VARS_PRESENT = [c for c in XGB_VARS if c in df_hurdle.columns]
print(f"Variables XGB : {len(XGB_VARS_PRESENT)}")

X_xgb_train = df_hurdle[XGB_VARS_PRESENT].fillna(0).values
y_xgb_train = df_hurdle['is_migration'].values

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_xgb_train == 0).sum() / (y_xgb_train == 1).sum(),
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(X_xgb_train, y_xgb_train)

auc_train = roc_auc_score(y_xgb_train, xgb_model.predict_proba(X_xgb_train)[:, 1])
print(f"XGB AUC train : {auc_train:.4f}")

proba_xgb_train = xgb_model.predict_proba(X_xgb_train)[:, 1].clip(eps, 1 - eps)
df_hurdle['logit_xgb'] = scipy_logit(proba_xgb_train)

XGB_VARS_TEST = [c for c in XGB_VARS_PRESENT if c in df_test.columns]
proba_xgb_test = xgb_model.predict_proba(df_test[XGB_VARS_TEST].fillna(0).values)[:, 1].clip(eps, 1 - eps)
df_test['logit_xgb'] = scipy_logit(proba_xgb_test)

HURDLE_VARS = HURDLE_VARS + ['logit_xgb']
K_h = len(HURDLE_VARS)
print(f"K_h : {K_h}")









###### ancien code Random Forest ######



# df_hurdle['log_gdpcap_diff'] = df_hurdle['log_gdpcap_d_lag'] - df_hurdle['log_gdpcap_o_lag']
# df_test['log_gdpcap_diff']   = df_test['log_gdpcap_d_lag']   - df_test['log_gdpcap_o_lag']
# RF_VARS = RF_VARS + ['log_gdpcap_diff']

# RF_VARS_PRESENT = [c for c in RF_VARS if c in df_hurdle.columns]
# print(f"Variables RF : {len(RF_VARS_PRESENT)}")

# eps = 1e-6
# X_rf_train = df_hurdle[RF_VARS_PRESENT].fillna(0).values
# y_rf_train = df_hurdle['is_migration'].values

# rf_model = RandomForestClassifier(
#     n_estimators=500, max_depth=10, min_samples_leaf=10,
#     class_weight='balanced', random_state=42, n_jobs=-1
# )
# rf_model.fit(X_rf_train, y_rf_train)

# auc_train = roc_auc_score(y_rf_train, rf_model.predict_proba(X_rf_train)[:, 1])
# print(f"RF AUC train : {auc_train:.4f}")

# proba_rf_train = rf_model.predict_proba(X_rf_train)[:, 1].clip(eps, 1 - eps)
# df_hurdle['logit_rf'] = scipy_logit(proba_rf_train)

# RF_VARS_TEST = [c for c in RF_VARS_PRESENT if c in df_test.columns]
# proba_rf_test = rf_model.predict_proba(df_test[RF_VARS_TEST].fillna(0).values)[:, 1].clip(eps, 1 - eps)
# df_test['logit_rf'] = scipy_logit(proba_rf_test)

# HURDLE_VARS = HURDLE_VARS + ['logit_rf']
# K_h = len(HURDLE_VARS)
# print(f"K_h : {K_h}")

# %%
# # grille XGBoost hurdle — comparaison AUC + FP/FN OOS 2015 par config


# #  Initialisation des paramètres

# PARAM_GRID = {
#     'max_depth':     [3, 4, 5, 6],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators':  [300, 500],
# }

# W_FP_global = 25.0


# cluster_test = df_test['continent_orig'].fillna(K_clusters).astype(int).values
# y_true       = df_test['flow'].values
# y_true_bin   = (y_true > 0).astype(int)

# # Poids pour compenser le déséquilibre des classes (51% de zéros) (???)
# scale_pos_weight = (y_xgb_train == 0).sum() / (y_xgb_train == 1).sum()

# results = []


# # Parcours de la grille de recherche

# for max_depth, lr, n_est in product(PARAM_GRID['max_depth'], PARAM_GRID['learning_rate'], PARAM_GRID['n_estimators']):

#     model = xgb.XGBClassifier(
#         n_estimators=n_est,
#         max_depth=max_depth,
#         learning_rate=lr,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         scale_pos_weight=scale_pos_weight,
#         eval_metric='logloss',
#         random_state=42,
#         n_jobs=-1,
#     )
    
#     model.fit(X_xgb_train, y_xgb_train)

#     proba_train = model.predict_proba(X_xgb_train)[:, 1]
#     proba_test  = model.predict_proba(df_test[XGB_VARS_TEST].fillna(0).values)[:, 1]

#     auc_train = roc_auc_score(y_xgb_train, proba_train)
#     prob_med  = proba_test.clip(1e-10, 1 - 1e-10)

    
#     # Optimisation régionale du seuil de décision
    
#     seuil_par_cluster = {}
#     for cluster_id in np.unique(cluster_test):
#         mask_c = (cluster_test == cluster_id)
#         n_pos  = y_true_bin[mask_c].sum()
#         n_neg  = (1 - y_true_bin[mask_c]).sum()
        
#         # Repli sur le seuil global si l'échantillon régional est insuffisant
#         if n_pos < 30 or n_neg < 30:
#             fpr_g, tpr_g, thresh_g = roc_curve(y_true_bin, prob_med)
#             seuil_par_cluster[cluster_id] = thresh_g[np.argmax(tpr_g - W_FP_global * fpr_g)]
#             continue
            
#         ratio        = n_neg / n_pos
#         ratio_global = (1 - y_true_bin).sum() / y_true_bin.sum()
#         wp_c         = np.clip(W_FP_global * (ratio / ratio_global), 2.0, 50.0)
        
#         fpr_c, tpr_c, thresh_c = roc_curve(y_true_bin[mask_c], prob_med[mask_c])
#         seuil_par_cluster[cluster_id] = thresh_c[np.argmax(tpr_c - wp_c * fpr_c)]

#     y_pred_bin = np.zeros(len(y_true_bin), dtype=int)
#     for cluster_id, seuil_c in seuil_par_cluster.items():
#         mask_c = (cluster_test == cluster_id)
#         y_pred_bin[mask_c] = (prob_med[mask_c] > seuil_c).astype(int)

    
#     #  Évaluation et stockage des métriques OOS
    
#     fp = ((y_pred_bin == 1) & (y_true_bin == 0)).sum()
#     fn = ((y_pred_bin == 0) & (y_true_bin == 1)).sum()
#     acc = (y_pred_bin == y_true_bin).mean()
#     auc_test = roc_auc_score(y_true_bin, prob_med)

#     results.append({
#         'max_depth': max_depth, 'lr': lr, 'n_estimators': n_est,
#         'auc_train': auc_train, 'auc_test': auc_test,
#         'acc': acc, 'fp': fp, 'fn': fn,
#     })

# # Synthèse
# df_results = pd.DataFrame(results).sort_values('auc_test', ascending=False)
# print(df_results.to_string(index=False))

# %% [markdown]
# La ligne max_depth=6, lr=0.05, n_estimators=500 offre le compromis biais-variance optimal. L'écart entre l'AUC Train (0,983582) et l'AUC Test (0,974432) est sain ($\Delta = 0,009$)

# %%
df_test['log_flow_lag_clean'] = (
    df_test['log_flow_lag'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
)

BINARY_COLS_VOL = ['LB_ij', 'OL_ij', 'COL_ij']
BINARY_COLS_HUR = ['LB_ij', 'COL_ij', 'OL_ij']

def standardize_matrix(X, col_names, binary_cols, fit_stats=None):
    X_std, stats = X.copy().astype(float), {}
    for j, col in enumerate(col_names):
        if col not in binary_cols:
            mu = X[:, j].mean() if fit_stats is None else fit_stats[col]['mean']
            sd = X[:, j].std()  if fit_stats is None else fit_stats[col]['std']
            sd = max(sd, 1e-8)
            X_std[:, j] = (X[:, j] - mu) / sd
            stats[col] = {'mean': mu, 'std': sd}
        else:
            stats[col] = {'mean': 0.0, 'std': 1.0}
    return X_std, stats

dyades_h = sorted(df_hurdle['dyad'].unique())
dyad_to_h = {d: i + 1 for i, d in enumerate(dyades_h)}
df_hurdle['dyad_id_h'] = df_hurdle['dyad'].map(dyad_to_h)
D_h = len(dyades_h)
cluster_h = (
    df_hurdle.groupby('dyad')['continent_orig'].first()
    .reindex([k for k, v in sorted(dyad_to_h.items(), key=lambda x: x[1])])
    .values.astype(int)
)

dyades_v = sorted(df_volume['dyad'].unique())
dyad_to_v = {d: i + 1 for i, d in enumerate(dyades_v)}
df_volume['dyad_id_v'] = df_volume['dyad'].map(dyad_to_v)
D_v = len(dyades_v)
cluster_v = (
    df_volume.groupby('dyad')['continent_orig'].first()
    .reindex([k for k, v in sorted(dyad_to_v.items(), key=lambda x: x[1])])
    .values.astype(int)
)

X_vol_std, stats_vol = standardize_matrix(df_volume[X_VOL_COLS].values, X_VOL_COLS, BINARY_COLS_VOL)
X_h_std,   stats_h   = standardize_matrix(df_hurdle[HURDLE_VARS].values, HURDLE_VARS, BINARY_COLS_HUR)

# %%
df_test['dyad'] = df_test['orig'] + "_" + df_test['dest']
df_test['dyad_id_test']   = df_test['dyad'].map(dyad_to_h)
df_test['dyad_id_test_v'] = df_test['dyad'].map(dyad_to_v).fillna(0).astype(int)
df_test = df_test.dropna(subset=['dyad_id_test']).copy().reset_index(drop=True)
df_test = df_test.dropna(subset=['log_gdpcap_d_lag'] + HURDLE_VARS + X_VOL_COLS).copy().reset_index(drop=True)

df_test['m49_brut'] = df_test['orig'].map(lambda x: ISO3_TO_M49_SUBREGION.get(str(x).upper(), 99))
df_test['continent_orig_fill'] = df_test['m49_brut'].map(_M49_TO_STAN).fillna(K_clusters).astype(int)
cluster_test_h = df_test['continent_orig_fill'].values

X_test_v_std, _ = standardize_matrix(df_test[X_VOL_COLS].values, X_VOL_COLS, BINARY_COLS_VOL, fit_stats=stats_vol)
X_test_h_std, _ = standardize_matrix(df_test[HURDLE_VARS].values, HURDLE_VARS, BINARY_COLS_HUR, fit_stats=stats_h)

tous_les_pays = sorted(list(set(df['orig'].unique()).union(set(df['dest'].unique()))))
pays_to_id    = {pays: i + 1 for i, pays in enumerate(tous_les_pays)}
N_pays_total  = len(tous_les_pays)

df_volume['orig_id_v'] = df_volume['orig'].map(pays_to_id)
df_volume['dest_id_v'] = df_volume['dest'].map(pays_to_id)
df_hurdle['orig_id_h'] = df_hurdle['orig'].map(pays_to_id)
df_hurdle['dest_id_h'] = df_hurdle['dest'].map(pays_to_id)
df_test['orig_id_test_v'] = df_test['orig'].map(pays_to_id)
df_test['dest_id_test_v'] = df_test['dest'].map(pays_to_id)

print(f"Test OOS : {len(df_test):,} obs")

# %%
K_Z = 2
Z_mat = np.zeros((N_pays_total, K_Z))

for pays, pays_id in pays_to_id.items():
    idx = pays_id - 1
    subset_orig = df_train[df_train['orig'] == pays]
    if not subset_orig.empty:
        pop = subset_orig['log_P_it'].mean()
        gdp = subset_orig['log_gdpcap_o_lag'].mean()
    else:
        subset_dest = df_train[df_train['dest'] == pays]
        pop = subset_dest['log_P_jt'].mean()
        gdp = subset_dest['log_gdpcap_d_lag'].mean()
    Z_mat[idx, 0] = pop
    Z_mat[idx, 1] = gdp

for j in range(K_Z):
    col_mean = np.nanmean(Z_mat[:, j])
    Z_mat[np.isnan(Z_mat[:, j]), j] = col_mean
    Z_mat[:, j] = (Z_mat[:, j] - np.mean(Z_mat[:, j])) / np.std(Z_mat[:, j])

Z_em = Z_mat
Z_at = Z_mat

# %%
df_hurdle = df_hurdle.replace([np.inf, -np.inf], np.nan).dropna(subset=HURDLE_REQUIRED)
df_volume = df_volume.replace([np.inf, -np.inf], np.nan).dropna(subset=VOLUME_REQUIRED)
assert np.isinf(df_volume[X_VOL_COLS].values).sum() == 0

stan_data = {
    'N_pays': N_pays_total,
    'N_h': int(N_h),
    'D_h': int(D_h),
    'K_h': int(K_h),
    'dyad_id_h': df_hurdle['dyad_id_h'].astype(int).tolist(),
    'is_mig': df_hurdle['is_migration'].astype(int).tolist(),
    'is_mig_lag': df_hurdle['is_mig_lag'].astype(float).tolist(),
    'X_h': X_h_std.tolist(),
    'cluster_h': cluster_h.tolist(),
    'K_Z': int(K_Z),
    'Z_em': Z_em.tolist(),
    'Z_at': Z_at.tolist(),
    'N_v': int(N_v),
    'D_v': int(D_v),
    'K_v': int(K_grav),
    'dyad_id_v': df_volume['dyad_id_v'].astype(int).tolist(),
    'orig_id_v': df_volume['orig_id_v'].astype(int).tolist(),
    'dest_id_v': df_volume['dest_id_v'].astype(int).tolist(),
    'flow': df_volume['flow'].astype(int).tolist(),
    'log_flow_lag': df_volume['log_flow_lag'].astype(float).tolist(),
    'X_v': X_vol_std.tolist(),
    'cluster_v': cluster_v.tolist(),
    'K_clusters': int(K_clusters),
    'do_ppc': 0,
    'do_loo': 0,
    'N_test': int(len(df_test)),
    'dyad_id_test_h': df_test['dyad_id_test'].astype(int).tolist(),
    'dyad_id_test_v': df_test['dyad_id_test_v'].astype(int).tolist(),
    'orig_id_test_v': df_test['orig_id_test_v'].astype(int).tolist(),
    'dest_id_test_v': df_test['dest_id_test_v'].astype(int).tolist(),
    'orig_id_h': df_hurdle['orig_id_h'].astype(int).tolist(),
    'dest_id_h': df_hurdle['dest_id_h'].astype(int).tolist(),
    'X_h_test': X_test_h_std.tolist(),
    'X_v_test': X_test_v_std.tolist(),
    'is_mig_lag_test': df_test['is_mig_lag'].fillna(0.0).tolist(),
    'log_flow_lag_test': df_test['log_flow_lag_clean'].tolist(),
    'cluster_test_h': cluster_test_h.tolist(),
}

# intégrité
anomalies = 0
for key, val in stan_data.items():
    if isinstance(val, (list, np.ndarray)):
        arr = np.array(val)
        if np.issubdtype(arr.dtype, np.number):
            if np.isnan(arr).sum() + np.isinf(arr).sum() > 0:
                print(f"[ERREUR] {key}")
                anomalies += 1
if anomalies == 0:
    print("stan_data : 0 NaN, 0 Inf")

# %%
binary = STAN_FILE.replace('.stan', '')
if os.path.exists(binary):
    os.remove(binary)

model = CmdStanModel(stan_file=STAN_FILE)

N_pays = df['orig'].nunique()

fit = model.sample(
    data            = stan_data,
    chains          = N_CHAINS,
    parallel_chains = PARALLEL_CHAINS,
    iter_warmup     = ITER_WARMUP,
    iter_sampling   = ITER_SAMPLING,
    save_warmup     = False,
    seed            = 42,
    inits           = 0.1,
    thin            = THIN,
    adapt_delta     = ADAPT_DELTA,
    max_treedepth   = MAX_TREEDEPTH,
    show_progress   = True,
    sig_figs        = 4,
    output_dir      = OUTPUT_DIR,
)

custom_prefix = f"ARX_{N_pays}pays_{N_CHAINS}c_{ITER_SAMPLING}it"
renamed_csvs = []
for i, old_path in enumerate(fit.runset.csv_files):
    new_path = os.path.join(OUTPUT_DIR, f"{custom_prefix}_chain{i+1}.csv")
    os.replace(old_path, new_path)
    renamed_csvs.append(new_path)

print(f"Outputs : {custom_prefix}_chain*.csv")
