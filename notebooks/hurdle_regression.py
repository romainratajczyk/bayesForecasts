#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cmdstanpy

cmdstan_base_dir = "/home/onyxia/work/cmdstan"

# 1. Détection et compilation conditionnelle du backend C++
if not os.path.exists(cmdstan_base_dir) or not any(d.startswith("cmdstan-") for d in os.listdir(cmdstan_base_dir)):
    print(f"Absence du backend C++. Téléchargement et compilation initiés dans {cmdstan_base_dir}...")
    # La compilation nécessite g++ et make (déjà présents sur l'image VSCode Python d'Onyxia)
    cmdstanpy.install_cmdstan(dir=cmdstan_base_dir)

# 2. Assignation stricte du chemin 
try:
    installed_versions = [d for d in os.listdir(cmdstan_base_dir) if d.startswith("cmdstan-")]
    # Trie pour garantir la sélection de la version la plus récente si plusieurs existent
    installed_versions.sort(reverse=True) 

    cmdstan_path = os.path.join(cmdstan_base_dir, installed_versions[0])
    cmdstanpy.set_cmdstan_path(cmdstan_path)
    print(f"Liaison matérielle CmdStan établie sur : {cmdstanpy.cmdstan_path()}")
except Exception as e:
    raise RuntimeError(f"Échec critique lors de la résolution des binaires CmdStan : {e}")



import psutil
import threading
import time
import numpy as np

class ClusterMonitor:
    def __init__(self, interval=2.0):
        """
        interval : fréquence d'échantillonnage en secondes.
        2.0s est optimal pour garantir un surcoût computationnel strictement nul.
        """
        self.interval = interval
        self.active = False
        self.ram_gi = []
        self.cpu_cores = []

    def _poll(self):
        main_proc = psutil.Process()
        main_proc.cpu_percent() # Initialisation des registres internes psutil

        while self.active:
            mem_bytes = 0
            cpu_pct = 0.0

            try:
                # Capture atomique de l'arbre des processus (Python + C++ CmdStan)
                procs = [main_proc] + main_proc.children(recursive=True)
                for p in procs:
                    try:
                        mem_bytes += p.memory_info().rss
                        # 100% CPU = 1 Coeur (Ci). interval=None pour calcul asynchrone.
                        cpu_pct += p.cpu_percent(interval=None)
                    except psutil.NoSuchProcess:
                        pass # Le processus C++ s'est terminé entre deux itérations

                self.ram_gi.append(mem_bytes / (1024**3))
                self.cpu_cores.append(cpu_pct / 100.0)
            except Exception:
                pass

            time.sleep(self.interval)

    def start(self):
        self.active = True
        self.t = threading.Thread(target=self._poll, daemon=True)
        self.t.start()

    def stop(self):
        self.active = False
        self.t.join()

        if not self.ram_gi:
            return

        ram_mean, ram_peak = np.mean(self.ram_gi), np.max(self.ram_gi)
        cpu_mean, cpu_peak = np.mean(self.cpu_cores), np.max(self.cpu_cores)

        print("\n" + "="*45)
        print(" DIAGNOSTIC MATÉRIEL (Onyxia) ".center(45))
        print("="*45)
        print(f" RAM (Gi)    | Moyenne : {ram_mean:>6.2f} | Pic : {ram_peak:>6.2f}")
        print(f" CPU (Cores) | Moyenne : {cpu_mean:>6.2f} | Pic : {cpu_peak:>6.2f}")
        print("="*45 + "\n")


# ## RHO PAR DYADE, MULTITHREADING, hurdle logit ~alpha + gamma + X_h*beta_h + beta_lag_m49[cluster_h]*is_mig_lag

# In[2]:


import warnings

import os
import cmdstanpy
from cmdstanpy import CmdStanModel

import pandas as pd
import numpy as np

import xgboost as xgb
import re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve , roc_auc_score, accuracy_score
from itertools import product

#from sklearn.ensemble import RandomForestClassifier
from scipy.special import logit as scipy_logit

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_PATH  = "../data/panel_june_filled.csv"
#STAN_FILE  = "../STAN/HMC_hurdle_regression_vectorized.stan"
OUTPUT_DIR = "./stan_outputs_tmux"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ## D'ici la réunion 8 Juillet 
# * réfléchir à un plan/draft; lire du JRSS A 
# * dataset: compléter ou réfléchir à exclure proprement les états problématiques. covariables concernées: LA LL urban ; PSE DROM-TOM etc.
# * relancer simulation avec Hurdle logit et rf / xgb, et quantifier rho dyadique/cluster 
# * résoudre faiblesse hyper-régression
# * trancher méthode W_FP
# * tester BART sur subsamples et trancher ; on peut le mentionner en Discussion. 
# 
# 
# ### tester gdpcap_lag1 pour l'horizon 1; réfléchir à la dimension temporelle (prédiction 2011-2014? =/= 2015?)
# 
# 
# 
# Var(flow_ij) = E[Var(flow | p_ij)] + Var(E[flow | p_ij])
#                     ^ variance volume          ^ variance hurdle
# ce deuxième terme est nul quand p_ij est donnée exogène du XGBoost

# In[ ]:


# Sampling parameters
N_CHAINS        = 4
PARALLEL_CHAINS = 4
ITER_WARMUP     = 500
ITER_SAMPLING   = 399
THIN            = 1
MAX_TREEDEPTH   = 12
ADAPT_DELTA     = 0.95
N_DRAWS         = ITER_SAMPLING // THIN

# Contrôle matériel : vectorized ou multithreading
USE_MULTITHREADING = False  # True (reduce_sum) / False (Vectorisation standard)


# SUBSET DE PAYS (modifier RUN_SIZE uniquement)

RUN_SIZE = 5
# _LABELS  = {1: '50 pays', 2: '80 pays', 3: '110 pays', 4: '140 pays', 5: '190 pays (complet)'}


# In[4]:


df_main = pd.read_csv(DATA_PATH)
df = df_main[df_main['orig'] != df_main['dest']].copy()

PAYS_EXCLURE = {
    'SSD', 'CUW', #'MNE', 'TLS', 
    'GUM', 'MYT', 'VIR', 'CLI', # on va essayer d'intégrer ceux-là manuellement (manque le PIB)
}
df = df[
    ~df['orig'].isin(PAYS_EXCLURE) &
    ~df['dest'].isin(PAYS_EXCLURE)
].copy()

df = df.sort_values(['orig', 'dest', 'year']).reset_index(drop=True)
print(f"{df['orig'].nunique()} pays après exclusions")


# In[5]:


# 1. Base 50 : Échantillon fondamental. Diversité géographique maximale et intégration des plus grands pôles.
PAYS_SUBSET_50 = {
    # Amérique du Nord & Sud
    'USA', 'CAN', 'MEX', 'BRA', 'ARG', 'COL', 'CHL', 'PER', 'VEN',
    # Europe
    'FRA', 'DEU', 'GBR', 'ITA', 'ESP', 'POL', 'RUS', 'UKR', 'SWE', 'NLD', 'ROU',
    # Asie & Moyen-Orient
    'CHN', 'IND', 'JPN', 'KOR', 'IDN', 'PAK', 'BGD', 'PHL', 'VNM', 'TUR', 'IRN', 'SAU', 'THA', 'MYS', 'KAZ',
    # Afrique
    'NGA', 'ETH', 'EGY', 'COD', 'ZAF', 'TZA', 'KEN', 'DZA', 'MAR', 'GHA', 'CIV', 'AGO', 'SEN',
    # Océanie
    'AUS', 'NZL'
}

# 2. Base 80 : +30 pays. Densification des régions d'Europe continentale, Asie centrale/sud, et Afrique subsaharienne.
PAYS_SUBSET_80 = PAYS_SUBSET_50 | {
    'NOR', 'FIN', 'DNK', 'CHE', 'AUT', 'BEL', 'GRC', 'CZE',
    'BOL', 'ECU', 'URY', 'GTM', 'CUB', 'DOM',
    'IRQ', 'ISR', 'ARE', 'UZB', 'MMR', 'LKA', 'NPL', 'AFG',
    'CMR', 'MLI', 'BFA', 'MOZ', 'ZMB', 'RWA', 'TUN', 'SDN'
}

# 3. Base 110 : +30 pays. Ajout de l'Europe de l'Est, Amérique centrale, péninsule arabique et Afrique francophone/australe.
PAYS_SUBSET_110 = PAYS_SUBSET_80 | {
    'HUN', 'PRT', 'IRL', 'BGR', 'SRB', 'HRV', 'BLR', 'SVK',
    'HND', 'SLV', 'NIC', 'CRI', 'PAN', 'PRY',
    'JOR', 'LBN', 'KWT', 'OMN', 'YEM', 'KHM', 'SGP',
    'TCD', 'NER', 'GIN', 'BDI', 'SOM', 'MWI', 'COG', 'GAB', 'NAM'
}

# 4. Base 140 : +30 pays. Ajout des pays baltes, Balkans, Caraïbes, Asie mineure et Océanie isolée.
PAYS_SUBSET_140 = PAYS_SUBSET_110 | {
    'LTU', 'LVA', 'EST', 'SVN', 'MKD', 'BIH', 'ALB', 'MDA',
    'HTI', 'JAM', 'TTO', 'BHS', 'GUY', 'SUR',
    'QAT', 'BHR', 'SYR', 'TJK', 'KGZ', 'LAO', 'MNG',
    'LBY', 'MRT', 'TGO', 'BEN', 'LBR', 'SLE', 'CAF',
    'PNG', 'FJI'
}


_SUBSETS = {1: PAYS_SUBSET_50, 2: PAYS_SUBSET_80, 3: PAYS_SUBSET_110, 4: PAYS_SUBSET_140, 5: None}
_LABELS  = {1: '50 pays', 2: '80 pays', 3: '110 pays', 4: '140 pays', 5: '190 pays (complet)'}


if RUN_SIZE < 5:
    pays_subset = _SUBSETS[RUN_SIZE]
    # Application des masques stricts sur orig et dest
    df = df[df['orig'].isin(pays_subset) & df['dest'].isin(pays_subset)].copy()
N_pays = df['orig'].nunique()
print(f"Run : {df['orig'].nunique()} pays dans le panel")


# In[6]:


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


# In[7]:


# df['is_migration'] = (df['flow'] > 0).astype(int)
# df['log_flow']     = np.where(df['flow'] > 0, np.log(df['flow']), np.nan)
# df['log_flow_lag'] = df.groupby(['orig', 'dest'])['log_flow'].shift(1)
# df['is_mig_lag']   = df.groupby(['orig', 'dest'])['is_migration'].shift(1)
# df['log_D_ij']     = np.log(df['D_ij'].replace(0, np.nan))
# df['log_D_ij_sq']  = df['log_D_ij'] ** 2
# df['logD_times_LB'] = df['log_D_ij'] * df['LB_ij']


df['dyad'] = df['orig'] + "_" + df['dest']

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


# In[8]:


# === Momentum du flux : Δ = log Y_{t-1} − log Y_{t-2} (pente, absente de l'ARX(1)) ===
df = df.sort_values(['dyad', 'year'])
lag1_raw = df.groupby('dyad')['log_flow'].shift(1)         # brut (NaN si fermé), PAS le patch
lag2_raw = df.groupby('dyad')['log_flow'].shift(2)
df['flow_momentum'] = (lag1_raw - lag2_raw).fillna(0.0)    # NaN -> 0 : pas de tendance observée
df = df.sort_values(['orig', 'dest', 'year']).reset_index(drop=True)
print(f"flow_momentum non-nul : {(df['flow_momentum'] != 0).mean()*100:.1f}% | "
      f"range [{df['flow_momentum'].min():.2f}, {df['flow_momentum'].max():.2f}]")


# In[9]:


# HURDLE_VARS RF avec colinéarité 
HURDLE_VARS = [
    'log_D_ij', 'log_D_ij_sq', 'COL_ij', 'OL_ij',
    'v2x_polyarchy_o_lag5',# 'v2x_clphy_o_lag1', 'intensity_level_o_lag1',
    'v2x_polyarchy_d_lag5'#, 'v2x_clphy_d_lag1', 'intensity_level_d_lag1'#, 'is_mig_lag'
]

X_VOL_COLS = [
    'log_D_ij', 'log_D_ij_sq', 'LB_ij', 'OL_ij', 'COL_ij', #'t_2000', 't_2000_sq',
    'v2x_polyarchy_o_lag5', 'intensity_level_o_lag5', #'v2x_clphy_o_lag5'
     'v2x_clphy_d_lag5', 'intensity_level_d_lag5'#, 'type_of_conflict_d_lag1', 'v2x_polyarchy_d_lag1'
]

# for lst_name in ['HURDLE_VARS', 'X_VOL_COLS']:
#     lst = globals()[lst_name]
#     if 'flow_momentum' not in lst:
#         globals()[lst_name] = lst + ['flow_momentum']


df_train = df[df['year'] <= 2010].copy()
#df_test = df[df['year'] == 2015].copy()
df_hold   = df[df['year'] == 2010].copy()      # calibration OOS
df_2015   = df[df['year'] == 2015].copy()
df_test   = pd.concat([df_hold, df_2015], ignore_index=True)
df_test['is_2015'] = (df_test['year'] == 2015).astype(int)
df_test_full = df_test.copy()
df_test_full['dyad'] = df_test_full['orig'] + "_" + df_test_full['dest']
df = df_train


# In[ ]:


HURDLE_REQUIRED = HURDLE_VARS + [ 'is_migration', 'dyad', 'continent_orig',
                                 'is_mig_lag'
                                 ] 
# covariables + is_mig_lag ne devant pas être standardisée et occupant une place théorique particulière (hystérésis) 
# + variables structurelles  dont Stan a besoin pour l'entraînement et la vraisemblance 
# (dyad pour les effets fixes alpha_i et gamma_j, continent_orig pour les effets de cluster M49)

# 1990-2005 uniquement pour vraie calibration OOS des FN/FP
df_hurdle = df[df['year'] <= 2005].dropna(subset=HURDLE_REQUIRED).copy().reset_index(drop=True)
#df_hurdle = df.dropna(subset=HURDLE_REQUIRED).copy().reset_index(drop=True)

df['has_history'] = (
    df.groupby(['orig', 'dest'])['flow']
    .transform(lambda x: (x.shift(1) > 0).expanding().max()) > 0
).astype(int) # voir notes .pages, history correspond à max(Y_0, Y_1, ..., Y_t-1)>0

#VOLUME_REQUIRED = X_VOL_COLS + ['flow', 'log_flow_lag', 'dyad', 'continent_orig']
VOLUME_REQUIRED = X_VOL_COLS + ['flow', 'is_mig_lag', 'has_history', 'dyad', 'continent_orig']
#df_volume = df[df['flow'] > 0].dropna(subset=VOLUME_REQUIRED).copy()
df_volume = df[df['flow'] > 0].dropna(subset=VOLUME_REQUIRED).copy().reset_index(drop=True)

# états markovien, pour handle amnésie markovienne dans le modèle de volume
is_continu = (df_volume['is_mig_lag'] == 1)
is_virgin  = (df_volume['is_mig_lag'] == 0) & (df_volume['has_history'] == 0) 
print(len(df_volume))
# EXCLUSION PURE des trous pour la ZTNB et le ARX(1)
is_censored = (df_volume['year'] == df_volume['year'].min())   # 1990 : pas de t-1

# Calcul des trous
is_reopen = (df_volume['is_mig_lag'] == 0) & (df_volume['has_history'] == 1)
n_trous = (is_reopen & ~is_censored).sum()

print(f"Nombre exact de trous exclus du ZTNB : {n_trous}")

df_volume = df_volume[(is_continu | is_virgin) & ~is_censored].copy().reset_index(drop=True)

print(f"Volume : {len(df_volume):,} obs (1990 exclu, pas de t-1 pour l'ARX)")

df_volume['is_emergent_v'] = (1 - df_volume['is_mig_lag']).astype(int)
df_volume['log_flow_lag_clean'] = df_volume['log_flow_lag'].fillna(0.0) # Bruit neutralisé par la bifurcation Stan

N_h, N_v = len(df_hurdle), len(df_volume)
print(f"Hurdle : {N_h:,} obs | Volume : {N_v:,} obs sans trous")


# In[11]:


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
df_test['transitivity_proxy'] = (df_test['out_degree_o'].fillna(0) * df_test['in_degree_d'].fillna(0))

# REVOIR CONSTRUCTION .MEAN DE TRANS_PROXY


#  A2_{t-1} : nb de chemins i->k->j actifs à t-1 (signal stepping-stone) 
# A[i,j] = is_mig_lag(i,j,t) = 1{Y_{ij,t-1} > 0}  ->  (A@A)[i,j] = #{k : i->k et k->j actifs}
pays_all = sorted(set(df_hurdle['orig']) | set(df_hurdle['dest'])
                  | set(df_test['orig']) | set(df_test['dest']))
pid = {p: i for i, p in enumerate(pays_all)}
Np = len(pays_all)

def a2_feature(frame):
    out = np.zeros(len(frame))
    for yr, sub in frame.groupby('year'):
        A = np.zeros((Np, Np), dtype=np.float32)
        act = sub[sub['is_mig_lag'] == 1]
        A[act['orig'].map(pid).values, act['dest'].map(pid).values] = 1.0
        A2 = A @ A                                   # ~192^3 flops : instantané
        oi = sub['orig'].map(pid).values
        dj = sub['dest'].map(pid).values
        out[frame.index.get_indexer(sub.index)] = A2[oi, dj]
    return out

df_hurdle['A2_log'] = np.log1p(a2_feature(df_hurdle))
df_test['A2_log']   = np.log1p(a2_feature(df_test))
df_volume['A2_log'] = np.log1p(a2_feature(df_volume))

if 'A2_log' not in HURDLE_VARS:
    HURDLE_VARS = HURDLE_VARS + ['A2_log']

#if 'A2_log' not in X_VOL_COLS: 
#    X_VOL_COLS = X_VOL_COLS + ['A2_log']

K_grav = len(X_VOL_COLS)
K_h = len(HURDLE_VARS) + 1 # +1 pour logit_xgb

# La démonstration clé : A^2 est non-nul là où toutes les variables d'inertie sont muettes
mask_fn_zone = (df_test['is_mig_lag'] == 0) & (df_test.get('log_stock_lag', 0) == 0)
print(f"Zone FN (lag=0, stock=0) : {mask_fn_zone.sum():,} dyades, "
      f"A2>0 pour {(df_test.loc[mask_fn_zone,'A2_log']>0).mean()*100:.1f}% d'entre elles")






# In[ ]:


from sklearn.ensemble import RandomForestClassifier

RF_VARS = [
    'log_D_ij', 'log_D_ij_sq', 'OL_ij', 'COL_ij',
    #'v2x_polyarchy_o_lag1', 'v2x_clphy_o_lag1', 'intensity_level_o_lag1',
    #'v2x_polyarchy_d_lag1', 'v2x_clphy_d_lag1', 'intensity_level_d_lag1',
    'log_gdpcap_o_lag5', 'log_gdpcap_d_lag5', #'log_gdpcap_diff',
    'log_P_it', 'log_P_jt',
    'is_mig_lag',
    'PSR_i', 'PSR_j',
    'IMR_it', 'IMR_jt',
    'urban_it', 'urban_jt',
    'LL_i', 'LL_j',
    'LA_i', 'LA_j',
    'LB_ij', #'logD_times_LB',
    #'type_of_conflict_o_lag5', #'type_of_conflict_d_lag5',
    'transitivity_proxy',
    'v2x_polyarchy_o_lag5', #'v2x_clphy_o_lag5', #'intensity_level_o_lag5', 'type_of_conflict_o_lag5',
    'v2x_polyarchy_d_lag5', #'v2x_clphy_d_lag5', #'intensity_level_d_lag5', 'type_of_conflict_d_lag5',
    'log_stock_lag', 'A2_log', 'flow_momentum'
    #'any_conflict_o_window', 'max_conflict_o_window', 'any_intense_o_window', 'any_intl_o_window',
    #'any_conflict_d_window', 'max_conflict_d_window', 'any_intense_d_window', 'any_intl_d_window',
    #'new_conflict_o', 'new_conflict_d', 'persistent_conflict_o', 'persistent_conflict_d', 
]

# for lst_name in ['RF_VARS']:
#     lst = globals()[lst_name]
#     if 'flow_momentum' not in lst:
#         globals()[lst_name] = lst + ['flow_momentum']

eps = 1e-6

RF_VARS_PRESENT = [c for c in RF_VARS if c in df_hurdle.columns]
print(f"Variables RF : {len(RF_VARS_PRESENT)}")

X_rf_train = df_hurdle[RF_VARS_PRESENT].fillna(0).values
y_rf_train = df_hurdle['is_migration'].values

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=10,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    oob_score=True
)
rf_model.fit(X_rf_train, y_rf_train)

auc_train = roc_auc_score(y_rf_train, rf_model.predict_proba(X_rf_train)[:, 1])
print(f"RF AUC train : {auc_train:.4f}")

# Injection du logit RF comme covariable de la régression Hurdle (modèle n°1)
proba_rf_train = rf_model.predict_proba(X_rf_train)[:, 1].clip(eps, 1 - eps)
df_hurdle['logit_rf'] = scipy_logit(proba_rf_train)

RF_VARS_TEST = [c for c in RF_VARS_PRESENT if c in df_test.columns]
proba_rf_test = rf_model.predict_proba(df_test[RF_VARS_TEST].fillna(0).values)[:, 1].clip(eps, 1 - eps)
df_test['proba_rf'] = proba_rf_test
df_test['logit_rf'] = scipy_logit(proba_rf_test)

HURDLE_VARS = HURDLE_VARS + ['logit_rf']
K_h = len(HURDLE_VARS)
print(f"K_h : {K_h}")


print(pd.Series(rf_model.feature_importances_, index=RF_VARS_PRESENT).sort_values(ascending=False).head(36).round(5))


# In[13]:


# /!!!!!\

df_test['log_flow_lag_clean'] = (
    df_test['log_flow_lag'].fillna(0.0).replace([np.inf, -np.inf], 0.0)
)

# /!!!!!\




BINARY_COLS_VOL = ['LB_ij', 'OL_ij', 'COL_ij']
BINARY_COLS_HUR = ['LB_ij', 'COL_ij', 'OL_ij','logit_rf'] # standardisation de logit_rf / xgb

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


# In[14]:


df_test['dyad'] = df_test['orig'] + "_" + df_test['dest']
#df_test['dyad_id_test']   = df_test['dyad'].map(dyad_to_h)
df_test['dyad_id_test']   = df_test['dyad'].map(dyad_to_h).fillna(1).astype(int)   # >=1 obligatoire
df_test['dyad_id_test_v'] = df_test['dyad'].map(dyad_to_v).fillna(0).astype(int)   # 0 = OK (branche prévue)
#df_test = df_test.dropna(subset=['dyad_id_test']).copy().reset_index(drop=True)
df_test = df_test.dropna(subset=['log_gdpcap_d_lag'] + HURDLE_VARS + X_VOL_COLS).copy().reset_index(drop=True)
df_test['dyad_id_test'] = df_test['dyad_id_test'].astype(int)

df_test['m49_brut'] = df_test['orig'].map(lambda x: ISO3_TO_M49_SUBREGION.get(str(x).upper(), 99))
df_test['continent_orig_fill'] = df_test['m49_brut'].map(_M49_TO_STAN).fillna(K_clusters).astype(int)
cluster_test_h = df_test['continent_orig_fill'].values  # toujours nécessaire pour rho_m49[k] et phi_disp_cluster[k] dans Stan


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


# In[15]:


K_Z = 1
Z_mat = np.zeros((N_pays_total, K_Z))

# Dernière période d'entraînement disponible par pays
df_last = df_train[df_train['year'] == df_train['year'].max()]  # year == 2010
#df_last = df_test[df_test['year'] == 2015]  

for pays, pays_id in pays_to_id.items():
    idx = pays_id - 1

    sub_orig = df_last[df_last['orig'] == pays]
    sub_dest = df_last[df_last['dest'] == pays]

    if not sub_orig.empty:
        log_mass = sub_orig['log_P_it'].iloc[0] + sub_orig['log_gdpcap_o_lag1'].iloc[0]
    elif not sub_dest.empty:
        log_mass = sub_dest['log_P_jt'].iloc[0] + sub_dest['log_gdpcap_d_lag1'].iloc[0]
    else:
        log_mass = np.nan

    Z_mat[idx, 0] = log_mass

col_median = np.nanmedian(Z_mat[:, 0])
for idx, pays in enumerate(pays_to_id.keys()):
    if np.isnan(Z_mat[idx, 0]):
        m49 = ISO3_TO_M49_SUBREGION.get(pays, 99)
        pays_meme_region = [p for p, m in ISO3_TO_M49_SUBREGION.items()
                            if m == m49 and p in pays_to_id]
        vals_region = [Z_mat[pays_to_id[p] - 1, 0]
                       for p in pays_meme_region
                       if not np.isnan(Z_mat[pays_to_id[p] - 1, 0])]
        Z_mat[idx, 0] = np.median(vals_region) if vals_region else col_median
Z_mat[:, 0] = (Z_mat[:, 0] - np.mean(Z_mat[:, 0])) / np.std(Z_mat[:, 0])

Z_em = Z_mat
Z_at = Z_mat

print(f"Z_mat — NaN résiduels : {np.isnan(Z_mat).sum()} | min : {Z_mat.min():.2f} | max : {Z_mat.max():.2f}")


# # penser à prendre le PIB courant de 2010 une fois dispo dans le dataset

# In[16]:


df_hurdle = df_hurdle.replace([np.inf, -np.inf], np.nan).dropna(subset=HURDLE_REQUIRED)
df_volume = df_volume.replace([np.inf, -np.inf], np.nan).dropna(subset=VOLUME_REQUIRED)
N_h, N_v = len(df_hurdle), len(df_volume)
assert np.isinf(df_volume[X_VOL_COLS].values).sum() == 0

_sc = df_volume.groupby('dyad')['log_flow_lag_clean'].mean().reindex(dyades_v).values
stan_data_extra_scale = (_sc - _sc.mean()) / max(_sc.std(), 1e-8)


stan_data = {
    # Dimensions globales 
    'N_pays'     : N_pays_total,
    'K_clusters' : int(K_clusters),

    # Hyper-régression 
    'K_Z'  : int(K_Z),
    'Z_em' : Z_em.tolist(),
    'Z_at' : Z_at.tolist(),

    # URDLE (train)
    'N_h'        : int(N_h),
    'D_h'        : int(D_h),
    'K_h'        : int(K_h),
    'dyad_id_h'  : df_hurdle['dyad_id_h'].astype(int).tolist(),
    'orig_id_h'  : df_hurdle['orig_id_h'].astype(int).tolist(),
    'dest_id_h'  : df_hurdle['dest_id_h'].astype(int).tolist(),
    'is_mig'     : df_hurdle['is_migration'].astype(int).tolist(),
    'is_mig_lag' : df_hurdle['is_mig_lag'].astype(float).tolist(),
    'X_h'        : X_h_std.tolist(),
    'cluster_h'  : cluster_h.tolist(),

    #  VOLUME (train) 
    'N_v'          : int(N_v),
    'D_v'          : int(D_v),
    'K_v'          : int(K_grav),
    'dyad_id_v'    : df_volume['dyad_id_v'].astype(int).tolist(),
    'orig_id_v'    : df_volume['orig_id_v'].astype(int).tolist(),
    'dest_id_v'    : df_volume['dest_id_v'].astype(int).tolist(),
    'flow'         : df_volume['flow'].astype(int).tolist(),
    #'log_flow_lag' : df_volume['log_flow_lag'].astype(float).tolist(),
    'is_emergent_v'   : df_volume['is_emergent_v'].astype(int).tolist(),          
    'log_flow_lag'    : df_volume['log_flow_lag_clean'].astype(float).tolist(),
    'X_v'          : X_vol_std.tolist(),
    'cluster_v'    : cluster_v.tolist(),
    'momentum_v'    : df_volume['flow_momentum'].astype(float).tolist(),
    'momentum_test' : df_test['flow_momentum'].astype(float).tolist(),

    'log_scale_v' : stan_data_extra_scale.tolist(),

    # TEST OOS — commun
    'N_test'         : int(len(df_test)),
    'cluster_test_h' : cluster_test_h.tolist(),

    # TEST OOS — Hurdle 
    'dyad_id_test_h'  : df_test['dyad_id_test'].astype(int).tolist(),
    'X_h_test'        : X_test_h_std.tolist(),
    'is_mig_lag_test' : df_test['is_mig_lag'].fillna(0.0).tolist(),

    # TEST OOS — Volume
    'dyad_id_test_v'    : df_test['dyad_id_test_v'].astype(int).tolist(),
    'orig_id_test_v'    : df_test['orig_id_test_v'].astype(int).tolist(),
    'dest_id_test_v'    : df_test['dest_id_test_v'].astype(int).tolist(),
    'X_v_test'          : X_test_v_std.tolist(),
    'log_flow_lag_test' : df_test['log_flow_lag_clean'].tolist(),

    # Flags 
    'do_ppc' : 0,
    'do_loo' : 0,
}

YEARS_CALIB = [int(df_train['year'].max()), int(df_train['year'].max() - 5)]  # 2010, 2005
calib_pos = np.where(df_hurdle['year'].isin(YEARS_CALIB).values)[0] + 1       # 1-based
stan_data.update({'N_calib': int(len(calib_pos)), 'calib_idx': calib_pos.tolist()})
assert X_h_std.shape[0] == len(df_hurdle), "X_h_std désynchronisé du dropna de [15]"


# Contrôle d'intégrité 
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


# In[17]:


#  Graphe de contiguïté pour le prior spatial (liste d'arêtes pour Stan)
lb_pairs = df_train[df_train['LB_ij'] == 1][['orig', 'dest']].drop_duplicates()
edges = set()
for o, d in zip(lb_pairs['orig'], lb_pairs['dest']):
    if o in pays_to_id and d in pays_to_id:
        i, j = pays_to_id[o], pays_to_id[d]
        if i != j:
            edges.add((min(i, j), max(i, j)))   # arêtes non-orientées, dédupliquées
edges = sorted(edges)
node1 = [e[0] for e in edges]
node2 = [e[1] for e in edges]

deg = np.zeros(N_pays_total + 1, dtype=int)
for i, j in edges:
    deg[i] += 1; deg[j] += 1
singletons = [p for p, pid in pays_to_id.items() if deg[pid] == 0]
print(f"Arêtes frontière : {len(edges)} | pays sans voisin terrestre (îles) : {len(singletons)}")

stan_data.update({'N_edges': len(edges), 'node1': node1, 'node2': node2})


# In[18]:


import numpy as np
import pandas as pd
from scipy import stats

def audit_paire(df, var1, var2, target='is_migration'):
    """relation entre deux variables """
    sub = df[[var1, var2, target]].replace([np.inf, -np.inf], np.nan).dropna()
    x, y, t = sub[var1].values, sub[var2].values, sub[target].values

    pearson_r, p_pear = stats.pearsonr(x, y)      # corrélation linéaire
    spearman_r, p_spear = stats.spearmanr(x, y)   # corrélation monotone (rangs)

    print(f" {var1}  vs  {var2} ")
    print(f"Pearson  r = {pearson_r:+.3f} (p={p_pear:.1e})  (linéaire)")
    print(f"Spearman r = {spearman_r:+.3f} (p={p_spear:.1e})   (monotone)")
    if abs(pearson_r) > 0.7:
        print(f"  warning; STRONG CORRELATION (|r|>0.7). Risk of instability")
    elif abs(pearson_r) > 0.5:
        print(f"  ~ Weak colinearity, attention")

    # Corrélation de chaque variable avec is_mig
    r1 = stats.pointbiserialr(t, x)[0]
    r2 = stats.pointbiserialr(t, y)[0]
    print(f"Corr({var1}, {target}) = {r1:+.3f}")
    print(f"Corr({var2}, {target}) = {r2:+.3f}")
    print()


audit_paire(df_volume, 'log_flow_lag', 'flow_momentum')
# audit_paire(df_hurdle, 'A2_log', 'transitivity_proxy')


# In[19]:


from sklearn.linear_model import LinearRegression

def vif_table(df, variables):
    """VIF : detect multivariate correlation. VIF>10 = real issue, >5 = attention."""
    sub = df[variables].replace([np.inf, -np.inf], np.nan).dropna()
    X = ((sub - sub.mean()) / sub.std()).values
    rows = []
    for j, var in enumerate(variables):
        others = [k for k in range(len(variables)) if k != j]
        r2 = LinearRegression().fit(X[:, others], X[:, j]).score(X[:, others], X[:, j])
        vif = 1.0 / (1.0 - r2) if r2 < 1 else np.inf
        flag = ' /!\\ ' if vif > 10 else (' ~ ' if vif > 5 else '')
        rows.append({'variable': var, 'VIF': round(vif, 2), 'R2_others': round(r2, 3), 'flag': flag})
    return pd.DataFrame(rows).sort_values('VIF', ascending=False)

conflict_block = ['intensity_level_d_lag1', 'type_of_conflict_d_lag1',
                  'v2x_polyarchy_d_lag1', 'v2x_clphy_d_lag1',
                  'v2x_polyarchy_o_lag1', 'v2x_clphy_o_lag1', 'intensity_level_o_lag1']
print(vif_table(df_volume, conflict_block))


# ### "Spatial Durbin–Watson"
# $$t_{DW} = n \cdot \frac{\sum_{a}\sum_{b} w_{ab}\,e_a e_b}{\sum_{a,b} w_{ab}\;\sum_a e_a^2}$$
# - $e_a$: residual of dyad $a$ (observed − predicted).
# - $w_{ab}=1$ if corridors $a,b$ are neighbours (share origin, share destination,
#   or origins share a land border). Choosing $W$ = choosing which DW we run.
# - **H0**: residuals are spatially independent.
# - **H1**: spatial autocorrelation ($t_{DW}>0$) — neighbour errors are correlated
#   => a spatial information is not captured yet.
# 
# Durbin–Watson in space instead of time. correlation between each residual and the average residual of the neighbours. Different definitions of W to explore different scales. 
# 
# ### Country emission effect: current model vs spatial generalisation
# 
# **Current (independent country residuals):**
# $$\alpha_i = Z_i\theta + \tau_\alpha\,\alpha_{\text{raw},i},
# \qquad \alpha_{\text{raw},i}\sim\mathcal N(0,1)
# \qquad\Longleftrightarrow\qquad
# \alpha_i \sim \mathcal N\!\big(Z_i\theta,\ \tau_\alpha\big)$$
# Fundamentals ($Z=\log$ GDP × pop) + one idiosyncratic term, independent per country.
# 
# 
# the $\mathcal N(0,1)$ iid prior is not very informative (desirable in general) but it is actually not neutral, it fix independent country effects. Our spatial test rejects this. We fix a wrong independence assumption. 
# 
# 
# Instead of saying "each country effect is independent, N(0,1)", we should say "each country effect resembles the average of its neighbours"
# 
# ### current model : $\alpha_i \sim \mathcal N\!\big(Z_i\theta,\ \tau_\alpha\big)$
# ### spatial AR model : $\alpha_i \sim \mathcal N\!\big(Z_i\theta + \bar\alpha_{\text{voisins}(i)},\ \tau_\alpha\big)$
# 
# ARX(1) in time pulls toward the past; spatial AR pulls toward neighbourhood 
# 
# # Remarks:
# 
# 
# 
# If a whole region is biased the same way (systematic FN errors in Latin America for instance, probably because our model is blind to MERCOSUR agreements), spatial AR will not fix it. It is the role of new network features ($A^2$, where A is the $192 \times 192$ matrix with 1 if there is an active corridor from i to j at t−1), which inject new spatial signal.
# 
# We can imagine a refinement of $A^2$ : multiscaling generalization, on paths of length > 2 ? 
# 
# summary of network features ideas: 
# - $A^2$ and their refinements
# - pull effect: is $j$ a new attracting hub ? (it is highly likely that it will open new corridors in the future)
# - push effect (same idea, but at the origin) 
# - 
# 
# 
# 
# 
# 
# ### transitivity proxy :
# out_degree_o = nombre de corridors sortants actifs depuis i  (somme de la ligne i de A) 
# in_degree_d  = nombre de corridors entrants actifs vers j  (somme de la colonne j de A) 
# transitivity_proxy = out_degree_o × in_degree_d
# 
# 
# ### data leakage 
# on thresholds for Hurdle opening decisions: we will implement a 2-fold calibration on the last training period, and we will test stationnarity (results and MAPE should not change between different period of calibration?)
# Calibration: train/test 50/50 or 70/30 or 80/20 ? 
# 
# 
# ### BART : 
# we tested it superfically, it seems that an addition of weak trees is not sufficient to capture complexity of interactions between migration variables. But we used basic parameters for $\alpha=0.95$ and $\beta=2$ to penalize for depth : $\alpha / (1+d)^{\beta}$, it only accept depth of 2 or 3 at maximum. Should we test $\beta = 0.5$ for instance ?  
# 
# 
# 
# ### variables:
# 
# see tests. 
# UN DESA: the number of people born in country i and living in country j
# 
# VIF (Variance Inflation Factor): 
# if a variable is well predicted by the others (R2 close to 1), it is redundant
# 
# 

# In[20]:


print("dyad_id_test_h : min =", df_test['dyad_id_test'].min(),
      "| hors-train =", int((df_test['dyad'].map(dyad_to_h).isna()).sum()))
print("dyad_id_test_v : min =", df_test['dyad_id_test_v'].min(),
      "| d_v=0 =", int((df_test['dyad_id_test_v'] == 0).sum()))
print("N_test =", len(df_test), "| dont 2010 :", int((df_test['year'] == 2010).sum()),
      "| dont 2015 :", int((df_test['year'] == 2015).sum()))


# In[21]:


assert min(stan_data['dyad_id_test_h']) >= 1, "dyad_id_test_h contient 0 -> Stan refusera"
assert min(stan_data['dyad_id_test_v']) >= 0
assert len(stan_data['X_h_test']) == stan_data['N_test'] == len(df_test)
assert len(stan_data['log_scale_v']) == D_v
print("OK, prêt pour Stan")


# In[22]:


print("D_v =", D_v, "| dyades test hors df_volume :",
      df_test.loc[df_test['dyad_id_test_v'] == 0, 'dyad'].nunique())
print("dont ouvertes en 2010 :",
      int(((df_test['dyad_id_test_v'] == 0) & (df_test['year'] == 2010) &
           (df_test['flow'] > 0)).sum()))


# In[ ]:


if USE_MULTITHREADING:
    STAN_FILE = "../STAN/HMC_hurdle_regression_multithread.stan"
    THREADS_PER_CHAIN = 4  # Saturation des 12 P-cores (3 threads * 4 chaînes)
else:
    STAN_FILE = "../STAN/HMC_hurdle_regression_vectorized_v3.stan"


# Purge du binaire précédent pour prévenir la corruption du cache compilateur
binary = STAN_FILE.replace('.stan', '')
if os.path.exists(binary):
    os.remove(binary)

# 1. Compilation Asymétrique
if USE_MULTITHREADING:
    model = CmdStanModel(
        stan_file=STAN_FILE,
        cpp_options={'STAN_THREADS': 'true'}
    )
    sample_kwargs = {'threads_per_chain': THREADS_PER_CHAIN}
    arch_suffix = "MT"
else:
    model = CmdStanModel(
        stan_file=STAN_FILE,
        force_compile=True,
        #cpp_options={'STAN_CPP_OPTIMS': 'true'} 
    )
    sample_kwargs = {} 
    arch_suffix = "VECT"

N_pays = df['orig'].nunique()

monitor = ClusterMonitor(interval=2.0)
monitor.start()


# Échantillonnage HMC-NUTS
fit = model.sample(
    data              = stan_data,
    chains            = N_CHAINS,
    parallel_chains   = PARALLEL_CHAINS,
    iter_warmup       = ITER_WARMUP,
    iter_sampling     = ITER_SAMPLING,
    save_warmup       = False,
    seed              = 42,
    inits             = 0.1,
    thin              = THIN,
    adapt_delta       = ADAPT_DELTA,
    max_treedepth     = MAX_TREEDEPTH,
    show_progress     = True,
    sig_figs          = 4,
    output_dir        = OUTPUT_DIR,
    **sample_kwargs   # Déballage dynamique : injecte threads_per_chain uniquement si défini
)

monitor.stop()

# Traçabilité des logs
custom_prefix = f"ARX_{N_pays}pays_{N_CHAINS}c_{ITER_SAMPLING}it_{arch_suffix}"
renamed_csvs = []
for i, old_path in enumerate(fit.runset.csv_files):
    new_path = os.path.join(OUTPUT_DIR, f"{custom_prefix}_chain{i+1}.csv")
    os.replace(old_path, new_path)
    renamed_csvs.append(new_path)

print(f"Outputs : {custom_prefix}_chain*.csv")


# In[290]:


# import cmdstanpy
# cmdstanpy.rebuild_cmdstan()   # long : plusieurs minutes


# # à faire rho_m49_draws = df_final.filter(regex=r'^rho_m49\.\d+$').values

# In[23]:


CSV_PREFIX    = f"ARX_{N_pays}pays_{N_CHAINS}c_{ITER_SAMPLING}it_{arch_suffix}"
csv_files = [
    f"{OUTPUT_DIR}/{CSV_PREFIX}_chain{i+1}.csv"
    for i in range(N_CHAINS)
]

csv_files = ['/Users/rratajczyk/Desktop/bayesForecasts/bayesTemp/onyxia data/stan_outputs_tmux/stan_outputs_tmux/ARX_190pays_4c_402it_VECT_chain1.csv',
             '/Users/rratajczyk/Desktop/bayesForecasts/bayesTemp/onyxia data/stan_outputs_tmux/stan_outputs_tmux/ARX_190pays_4c_402it_VECT_chain2.csv',
             '/Users/rratajczyk/Desktop/bayesForecasts/bayesTemp/onyxia data/stan_outputs_tmux/stan_outputs_tmux/ARX_190pays_4c_402it_VECT_chain3.csv',
             '/Users/rratajczyk/Desktop/bayesForecasts/bayesTemp/onyxia data/stan_outputs_tmux/stan_outputs_tmux/ARX_190pays_4c_402it_VECT_chain4.csv']
with open(csv_files[0], 'r') as f:
    for line in f:
        if not line.startswith('#'):
            all_cols = line.strip().split(',')
            break

vars_to_keep = [
    #  Volume 
    'mu_dt_test', 'phi_test', 'omega',
    'beta_grav', 'phi_disp_global', 'phi_disp_cluster',
    'rho_global_monitor', 'rho_m49', 'sigma_rho_m49', 'tau_rho', 'tau_phi_disp', 'rho_m49_lat', # échelle LATENTE
    'tau_em', 'tau_at', 'intercept_em', 'intercept_at', 'theta_em', 'theta_at',
    'alpha_em', 'gamma_at',
    #  Hurdle (coef A2, Moran post-run, effet du champ spatial) 
    'prob_mig_test',
    'beta_h', 'beta_lag_m49', 'mu_beta_lag', 'sigma_beta_lag',
    'intercept_h_em', 'intercept_h_at', 'theta_h_em', 'theta_h_at',
    'tau_h_em', 'tau_h_at', 'alpha_h_em', 'gamma_h_at',
    'u_em', 'tau_u_em',
    #  Sampler 
    'divergent__', 'treedepth__', 'energy__', 'stepsize__',
]

# match exact OU préfixe pointé (évite d'aspirer alpha_em_raw via 'alpha_em')
cols_keep = [c for c in all_cols
             if any(c == v or c.startswith(v + '.') for v in vars_to_keep)]
print(f"Colonnes extraites : {len(cols_keep)}")

dfs = []
for f in csv_files:
    print(f"Lecture {f}...")
    dfs.append(pd.read_csv(f, comment='#', usecols=cols_keep, engine='c'))

df_final = pd.concat(dfs, ignore_index=True)
del dfs
print(f"RAM : {df_final.memory_usage().sum() / 1024**2:.1f} Mo")


# In[24]:


# YEARS_CALIB = [df_train['year'].max(), df_train['year'].max()-5]
# pos = {yr: np.where(df_hurdle['year'].values == yr)[0] for yr in YEARS_CALIB}
# keep_j = set(np.concatenate([pos[y] for y in YEARS_CALIB]) + 1)   # +1 : Stan indexe à partir de 1

with open(csv_files[0]) as f:
    for line in f:
        if not line.startswith('#'):
            header = line.strip().split(','); break



ph_all  = {c: int(c.split('.')[1]) for c in header if c.startswith('p_hurdle.')}
ph_cols = [c for c in header if c.startswith('p_hurdle.')]          # p_hurdle 2010+2005 seulement
assert len(ph_cols) == len(calib_pos), "désync calib_idx / colonnes p_hurdle"
pmt_cols = [c for c in header if c.startswith('prob_mig_test.')]  # tout le test 2015
usecols = ph_cols + pmt_cols
print(f"p_hurdle (2010+2005) : {len(ph_cols)} col | prob_mig_test : {len(pmt_cols)} col")

# moyenne posterior par observation, accumulée sur les chains (pas de concat en RAM)
acc, ndraw = None, 0
for fpath in csv_files:
    ch = pd.read_csv(fpath, comment='#', usecols=usecols, engine='c')
    acc = ch.sum() if acc is None else acc + ch.sum()
    ndraw += len(ch)
post = acc / ndraw       # Series : nom_colonne -> moyenne posterior

# remap p_hurdle -> df_hurdle (positions 2010 et 2005)
ph_vec = np.full(len(df_hurdle), np.nan)
for c in ph_cols:
    j = int(c.split('.')[1])
    ph_vec[calib_pos[j - 1] - 1] = post[c]
df_hurdle['p_hurdle'] = ph_vec


pmt_sorted = sorted(pmt_cols, key=lambda c: int(c.split('.')[1]))
df_test['p_hurdle'] = post[pmt_sorted].values
print(f"p_hurdle rempli : train 2010={df_hurdle.query('year==2010').p_hurdle.notna().sum():,}, "
      f"test 2015={df_test['p_hurdle'].notna().sum():,}")


# In[25]:


#prob_mig        = df_final.filter(like='prob_mig_test').values
mu_test         = df_final.filter(like='mu_dt_test').values
phi_t           = df_final.filter(like='phi_test').values
beta_grav       = df_final.filter(like='beta_grav').values
beta_h          = df_final.filter(like='beta_h').values
phi_disp_cluster = df_final.filter(like='phi_disp_cluster').values
rho_m49_draws = df_final.filter(regex=r'^rho_m49\.\d+$').values



print(f"mu_test shape : {mu_test.shape}")


# In[26]:


valid_draws = ~(
    np.isnan(mu_test).any(axis=1) |
    np.isnan(phi_t).any(axis=1)
)
mu_clean  = mu_test[valid_draws]
phi_clean = phi_t[valid_draws]
print(f"Tirages nettoyés : {mu_test.shape[0] - valid_draws.sum()} retirés")

eta_safe = np.clip(mu_clean, -50.0, 50.0)
phi_safe = np.clip(phi_clean, 1e-8, 1e8)
lam      = np.exp(eta_safe)
n_sp     = phi_safe
p_sp     = np.clip(phi_safe / (phi_safe + lam), 1e-10, 1.0 - 1e-10)

flow_cond_sim = np.random.negative_binomial(n_sp, p_sp)

zeros_mask    = (flow_cond_sim == 0)
max_retries, retries = 30, 0
while zeros_mask.any() and retries < max_retries:
    flow_cond_sim[zeros_mask] = np.random.negative_binomial(n_sp[zeros_mask], p_sp[zeros_mask])
    zeros_mask = (flow_cond_sim == 0)
    retries   += 1
if zeros_mask.any():
    flow_cond_sim[zeros_mask] = 1

flow_cond_med_final = np.median(flow_cond_sim, axis=0)
flow_cond_q25       = np.percentile(flow_cond_sim, 25, axis=0)

# Hurdle : XGB seul, pas de draws Stan
# prob_med = df_test['proba_rf'].values


# In[48]:


# ============================================================
# CALIBRATION DES SEUILS ET PRODUCTION — deux modes
#   MODE_CALIB = 'roc'   : Youden pondéré par cluster (W_FN, W_FP),
#                          n'utilise QUE le binaire → aucun flux prédit n'entre
#                          dans la calibration (robustesse, SM G)
#   MODE_CALIB = 'cout'  : perte asymétrique MAE + lambda*100*MAPE (papier)
# ============================================================
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve

# ------------------ CURSEURS ------------------
MODE_CALIB = 'cout'          # 'cout' | 'roc'
LAMBDA     = 3.0             # mode 'cout'
W_FN, W_FP = 1.0, 1.0        # mode 'roc' : 1.0/1.0 = Youden pur
MIN_CLUSTER = 30             # effectif minimal par classe pour un seuil local
# ----------------------------------------------

# ---------- 1. split ----------
m_cal = (df_test['is_2015'].values == 0)
m_ev  = (df_test['is_2015'].values == 1)
assert m_cal.sum() == m_ev.sum(), "split 2010/2015 déséquilibré"
df_ev = df_test.loc[m_ev].reset_index(drop=True)
mu_ev, phi_ev = mu_clean[:, m_ev], phi_clean[:, m_ev]
sim_ev = flow_cond_sim[:, m_ev]
flow_med_ev, flow_q25_ev = flow_cond_med_final[m_ev], flow_cond_q25[m_ev]
print(f"Calibration 2010 : {m_cal.sum():,} | Évaluation 2015 : {m_ev.sum():,}")

# ---------- 2. vecteurs ----------
p_cal = df_test.loc[m_cal, 'p_hurdle'].values
f_cal = df_test.loc[m_cal, 'flow'].values
c_cal = df_test.loc[m_cal, 'continent_orig_fill'].values
vol_cal = np.where(df_test.loc[m_cal, 'is_mig_lag'].fillna(0).values == 0,
                   flow_cond_q25[m_cal], flow_cond_med_final[m_cal])
y_cal_bin = (f_cal > 0).astype(int)

p_ev  = df_ev['p_hurdle'].values
c_ev  = df_ev['continent_orig_fill'].values
y_true, y_true_bin = df_ev['flow'].values, (df_ev['flow'].values > 0).astype(int)
vol_ev = np.where(df_ev['is_mig_lag'].fillna(0).values == 0, flow_q25_ev, flow_med_ev)

GRID = np.quantile(p_cal, np.linspace(0.05, 0.9995, 250))

# ---------- 3. calibration ----------
def seuils_cout(lam):
    """Perte asymétrique. Utilise vol_cal : dépend du volume prédit."""
    thr = {}
    for c in np.unique(c_cal):
        mc = (c_cal == c)
        f, v, p = f_cal[mc], vol_cal[mc], p_cal[mc]
        best_t, best_j = GRID[-1], np.inf
        for t in GRID:
            e = np.abs(f - np.where(p >= t, v, 0.0))
            j = e.sum() + lam * 100 * (e / (f + 1)).sum()
            if j < best_j: best_j, best_t = j, t
        thr[c] = float(best_t)
    return thr

def seuils_roc(w_fn=W_FN, w_fp=W_FP):
    """Youden pondéré. N'utilise QUE le binaire : aucun volume prédit."""
    fpr_g, tpr_g, thr_g = roc_curve(y_cal_bin, p_cal)
    t_global = float(thr_g[np.argmax(w_fn * tpr_g - w_fp * fpr_g)])
    thr = {}
    for c in np.unique(c_cal):
        mc = (c_cal == c)
        n_pos, n_neg = y_cal_bin[mc].sum(), (1 - y_cal_bin[mc]).sum()
        if n_pos < MIN_CLUSTER or n_neg < MIN_CLUSTER:
            thr[c] = t_global; continue
        fpr, tpr, t = roc_curve(y_cal_bin[mc], p_cal[mc])
        thr[c] = float(t[np.argmax(w_fn * tpr - w_fp * fpr)])
    return thr

def bilan(thr, p, c, f, v, t_def):
    pred = (p >= np.array([thr.get(k, t_def) for k in c])).astype(int)
    yb, yh = (f > 0).astype(int), np.where(pred == 1, v, 0.0)
    return (int(((pred==1)&(yb==0)).sum()), int(((pred==0)&(yb==1)).sum()),
            np.abs(f-yh).mean(), (np.abs(f-yh)/(f+1)).mean()*100, pred)

# ---------- 3bis. frontière ----------
if MODE_CALIB == 'cout':
    print(f"\n{'LAMBDA':>7} | {'CALIB 2010':^28} | {'ÉVAL 2015':^30}")
    print(f"{'':>7} | {'FP':>6} {'FN':>6} {'MAE':>6} {'MAPE':>6} | {'FP':>6} {'FN':>6} {'MAE':>7} {'MAPE':>6}")
    print("-" * 74)
    for lam in [0.0, 1.0, 3.0, 6.0, 10.0, 20.0]:
        th = seuils_cout(lam); td = float(np.median(list(th.values())))
        a = bilan(th, p_cal, c_cal, f_cal,  vol_cal, td)
        b = bilan(th, p_ev,  c_ev,  y_true, vol_ev,  td)
        print(f"{lam:>7.1f} | {a[0]:>6,} {a[1]:>6,} {a[2]:>6,.0f} {a[3]:>5.1f}% | "
              f"{b[0]:>6,} {b[1]:>6,} {b[2]:>7,.0f} {b[3]:>5.1f}%")
else:
    print(f"\n{'W_FP':>7} | {'ÉVAL 2015':^32}")
    print(f"{'':>7} | {'FP':>6} {'FN':>6} {'MAE':>7} {'MAPE':>6}")
    print("-" * 42)
    for w in [1.0, 2.0, 5.0, 10.0, 25.0]:
        th = seuils_roc(W_FN, w); td = float(np.median(list(th.values())))
        b = bilan(th, p_ev, c_ev, y_true, vol_ev, td)
        print(f"{w:>7.1f} | {b[0]:>6,} {b[1]:>6,} {b[2]:>7,.0f} {b[3]:>5.1f}%")

# ---------- 3ter. concordance des deux modes (SM G) ----------
th_c, th_r = seuils_cout(LAMBDA), seuils_roc()
ks = sorted(set(th_c) & set(th_r))
print(f"\nCorrélation seuils coût(λ={LAMBDA}) vs ROC(W={W_FN}/{W_FP}) : "
      f"{np.corrcoef([th_c[k] for k in ks], [th_r[k] for k in ks])[0,1]:.3f}")
for nom, th in [('coût', th_c), ('ROC', th_r)]:
    b = bilan(th, p_ev, c_ev, y_true, vol_ev, float(np.median(list(th.values()))))
    print(f"  {nom:<5} FP {b[0]:>5,} FN {b[1]:>5,} MAE {b[2]:>6,.0f} MAPE {b[3]:5.1f}%")

# ---------- 4. production ----------
thr_c = seuils_cout(LAMBDA) if MODE_CALIB == 'cout' else seuils_roc()
t_def = float(np.median(list(thr_c.values())))
fp, fn, _, _, y_pred_bin = bilan(thr_c, p_ev, c_ev, y_true, vol_ev, t_def)
tp = int(((y_pred_bin==1)&(y_true_bin==1)).sum())
print(f"\n[{MODE_CALIB}] OOS 2015 : FP={fp:,} | FN={fn:,} | TP={tp:,} | "
      f"précision={tp/max(tp+fp,1):.3f} | rappel={tp/max(tp+fn,1):.3f}")
print("\nSeuils par cluster :")
for c in sorted(thr_c):
    print(f"  {SUBREGION_LABELS.get(stan_to_m49.get(c,99), f'c{c}'):<22} t* = {thr_c[c]:.4f}")

# ---------- 5. prédictions et intervalles ----------
y_pred = np.where(y_pred_bin == 1, vol_ev, 0.0)
prob_draws = df_final.filter(like='prob_mig_test').values[valid_draws][:, m_ev]
is_mig_sim = np.random.binomial(1, np.clip(prob_draws, 0, 1))
flow_all   = is_mig_sim * sim_ev
y_pred_q05 = np.percentile(flow_all, 2.5,  axis=0)
y_pred_q95 = np.percentile(flow_all, 97.5, axis=0)

# ---------- 6. métriques ----------
global_mae = np.mean(np.abs(y_true - y_pred))
mape_wr    = np.mean(np.abs(y_true - y_pred) / (y_true + 1.0)) * 100
wmape      = np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-8) * 100
log_mae    = np.mean(np.abs(np.log1p(y_true) - np.log1p(y_pred)))
coverage_g = np.mean((y_true >= y_pred_q05) & (y_true <= y_pred_q95))

print(f"\nOuvertures prédites : {y_pred_bin.sum():,} / {len(y_pred_bin):,} "
      f"(réel : {y_true_bin.sum():,})")
print(f"Accuracy Hurdle : {accuracy_score(y_true_bin, y_pred_bin)*100:.1f}% | "
      f"WMAPE : {wmape:.1f}% | Log-MAE : {log_mae:.4f}")
print(f"\n{'Modèle':<34} | {'MAE':>9} | {'MAPE':>8} | {'Coverage':>9}")
print("-" * 70)
print(f"{'Welch & Raftery (2022)':<34} | {'~1,200':>9} | {'76.0%':>8} | {'93.0%':>9}")
print(f"{f'Hurdle ARX ZTNB ({N_pays} pays)':<34} | {global_mae:>9,.0f} | "
      f"{f'{mape_wr:.1f}%':>8} | {f'{coverage_g*100:.1f}%':>9}")

# ---------- 7. décomposition MAE ----------
fp_m = (y_pred_bin==1)&(y_true_bin==0); fn_m=(y_pred_bin==0)&(y_true_bin==1)
tp_m = (y_pred_bin==1)&(y_true_bin==1); n = len(y_true); err = np.abs(y_true - y_pred)
print(f"\npart FP : {y_pred[fp_m].sum()/n:,.1f} ({fp_m.sum()}) | "
      f"part FN : {y_true[fn_m].sum()/n:,.1f} ({fn_m.sum()}) | "
      f"part TP : {err[tp_m].sum()/n:,.1f} ({tp_m.sum()}) | "
      f"top-20 = {np.sort(err)[-20:].sum()/err.sum()*100:.0f}%")

# ---------- 8. couverture par sous-région M49 ----------
print(f"\n{'Sous-région':<24} {'n':>7} {'couverture':>11}")
print("-" * 45)
for c in sorted(np.unique(c_ev)):
    mc = (c_ev == c)
    cv = np.mean((y_true[mc] >= y_pred_q05[mc]) & (y_true[mc] <= y_pred_q95[mc]))
    print(f"{SUBREGION_LABELS.get(stan_to_m49.get(c,99), f'c{c}'):<24} "
          f"{mc.sum():>7,} {cv*100:>10.1f}%")


# In[51]:


for tag, mk in [('tous', np.ones(len(y_true), bool)),
                ('flux > 0', y_true_bin == 1), ('flux = 0', y_true_bin == 0)]:
    cv = np.mean((y_true[mk] >= y_pred_q05[mk]) & (y_true[mk] <= y_pred_q95[mk]))
    lg = np.mean(y_pred_q95[mk] - y_pred_q05[mk])
    print(f"{tag:<10} n={mk.sum():>6,}  couverture {cv*100:5.1f}%  largeur moy. {lg:>10,.0f}")


# In[ ]:


# # ============================================================
# # ABLATION HURDLE — BART / RF / XGB seuls vs logit bayésien
# # Train 1990-2005 (imposé par df_hurdle) | seuils calibrés sur 2010 | éval 2015
# # Balayage complet de lambda + couverture prédictive 95%
# # PLACEMENT : après la cellule de calibration de production
# # ============================================================
# import numpy as np, pandas as pd, time
# from itertools import product
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_auc_score, accuracy_score
# import xgboost as xgb

# # ------------------ PARAMÈTRES ------------------
# RUN_RF, RUN_XGB, RUN_BART = True, True, True
# LAMBDA_GRID    = [0.0, 1.0, 2.0, 3.0, 6.0, 10.0]
# BART_SUBSAMPLE = 40_000                      # None = 140k lignes
# BART_NDPOST, BART_NSKIP = 500, 250
# SEED_COV       = 0

# GRID_RF   = {'n_estimators': [300], 'max_depth': [20], 'min_samples_leaf': [10]}
# GRID_XGB  = {'max_depth': [8], 'learning_rate': [0.01], 'n_estimators': [800]}
# GRID_BART = [(200, 2.0, 2.0), (500, 2.0, 0.5)]      # (ntree, k, power)
# # ------------------------------------------------

# assert set(RF_VARS_PRESENT) == set(RF_VARS_TEST), \
#     f"colonnes absentes de df_test : {set(RF_VARS_PRESENT) - set(RF_VARS_TEST)}"

# X_tr = df_hurdle[RF_VARS_PRESENT].fillna(0).values
# y_tr = df_hurdle['is_migration'].values.astype(int)
# X_te = df_test[RF_VARS_TEST].fillna(0).values
# print(f"Train : {len(X_tr):,} obs, années {sorted(df_hurdle['year'].unique())}")

# m_cal = (df_test['is_2015'].values == 0)
# m_ev  = (df_test['is_2015'].values == 1)
# f_cal, c_cal = df_test.loc[m_cal,'flow'].values, df_test.loc[m_cal,'continent_orig_fill'].values
# y_true, c_ev = df_test.loc[m_ev, 'flow'].values, df_test.loc[m_ev, 'continent_orig_fill'].values
# y_cal_bin, y_true_bin = (f_cal > 0).astype(int), (y_true > 0).astype(int)

# vol_cal = np.where(df_test.loc[m_cal,'is_mig_lag'].fillna(0).values == 0,
#                    flow_cond_q25[m_cal], flow_cond_med_final[m_cal])
# vol_ev  = np.where(df_test.loc[m_ev, 'is_mig_lag'].fillna(0).values == 0,
#                    flow_cond_q25[m_ev],  flow_cond_med_final[m_ev])
# sim_ev  = flow_cond_sim[:, m_ev]                      # (S_vol, 35910)

# # ---------- métriques à lambda fixé ----------
# def metrics_lambda(p_all, lam):
#     p_c, p_e = p_all[m_cal], p_all[m_ev]
#     grid, thr = np.quantile(p_c, np.linspace(0.05, 0.9995, 250)), {}
#     for c in np.unique(c_cal):
#         mc = (c_cal == c)
#         f, v, p = f_cal[mc], vol_cal[mc], p_c[mc]
#         best_t, best_j = grid[-1], np.inf
#         for t in grid:
#             e = np.abs(f - np.where(p >= t, v, 0.0))
#             j = e.sum() + lam * 100 * (e / (f + 1)).sum()
#             if j < best_j: best_j, best_t = j, t
#         thr[c] = float(best_t)
#     t_def = float(np.median(list(thr.values())))
#     pred  = (p_e >= np.array([thr.get(k, t_def) for k in c_ev])).astype(int)
#     yh    = np.where(pred == 1, vol_ev, 0.0)
#     err   = np.abs(y_true - yh)
#     fp_m, fn_m = (pred==1)&(y_true_bin==0), (pred==0)&(y_true_bin==1)
#     tp_m = (pred==1)&(y_true_bin==1)
#     return {'lambda': lam, 'acc': accuracy_score(y_true_bin, pred),
#             'FP': int(fp_m.sum()), 'FN': int(fn_m.sum()),
#             'MAE': err.mean(), 'MAPE': (err/(y_true+1)).mean()*100,
#             'precision': tp_m.sum()/max(tp_m.sum()+fp_m.sum(), 1),
#             'rappel':    tp_m.sum()/max(tp_m.sum()+fn_m.sum(), 1)}

# # ---------- couverture prédictive 95% ----------
# def coverage(p_ev_point, p_ev_draws=None, seed=SEED_COV):
#     """plug-in : p ponctuel (ignore l'incertitude hurdle).
#        full    : tirages du posterior hurdle appariés aux tirages de volume."""
#     rng = np.random.RandomState(seed)
#     out = {}
#     S_v = sim_ev.shape[0]
#     z = rng.binomial(1, np.clip(np.tile(p_ev_point, (S_v, 1)), 0, 1))
#     fa = z * sim_ev
#     out['cov_plugin'] = np.mean((y_true >= np.percentile(fa, 2.5,  axis=0)) &
#                                 (y_true <= np.percentile(fa, 97.5, axis=0)))
#     if p_ev_draws is not None:
#         S_h = p_ev_draws.shape[0]
#         idx = rng.choice(S_v, S_h, replace=(S_h > S_v))
#         fa2 = rng.binomial(1, np.clip(p_ev_draws, 0, 1)) * sim_ev[idx]
#         out['cov_full'] = np.mean((y_true >= np.percentile(fa2, 2.5,  axis=0)) &
#                                   (y_true <= np.percentile(fa2, 97.5, axis=0)))
#     else:
#         out['cov_full'] = np.nan
#     return out

# MODELS = {}   # nom -> {'p': (71820,), 'draws_ev': (S,35910)|None, 'auc_tr': float, 'sec': float}

# # ---------- référence bayésienne ----------
# _ref_draws = df_final.filter(like='prob_mig_test').values[valid_draws][:, m_ev]
# MODELS["Hurdle bayésien"] = {'p': df_test['p_hurdle'].values,
#                              'draws_ev': _ref_draws, 'auc_tr': np.nan, 'sec': np.nan}

# # ---------- RANDOM FOREST ----------
# if RUN_RF:
#     for n_est, d, leaf in product(*GRID_RF.values()):
#         t0 = time.time()
#         m = RandomForestClassifier(n_estimators=n_est, max_depth=d, min_samples_leaf=leaf,
#                                    max_features='sqrt', class_weight='balanced',
#                                    oob_score=True, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
#         MODELS[f"RF n{n_est}_d{d}_leaf{leaf}"] = {
#             'p': m.predict_proba(X_te)[:, 1], 'draws_ev': None,
#             'auc_tr': roc_auc_score(y_tr, m.oob_decision_function_[:, 1]),  # OOB, pas in-sample
#             'sec': time.time()-t0}

# # ---------- XGBOOST ----------
# if RUN_XGB:
#     spw = (y_tr == 0).sum() / (y_tr == 1).sum()
#     for d, lr, n_est in product(*GRID_XGB.values()):
#         t0 = time.time()
#         m = xgb.XGBClassifier(n_estimators=n_est, max_depth=d, learning_rate=lr,
#                               subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
#                               eval_metric='logloss', random_state=42, n_jobs=-1).fit(X_tr, y_tr)
#         MODELS[f"XGB d{d}_lr{lr}_n{n_est}"] = {
#             'p': m.predict_proba(X_te)[:, 1], 'draws_ev': None,
#             'auc_tr': roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1]), 'sec': time.time()-t0}

# # ---------- BART ----------
# if RUN_BART:
#     import rpy2.robjects as ro
#     from rpy2.robjects import numpy2ri
#     from rpy2.robjects.conversion import localconverter
#     ro.r('library(dbarts)')

#     if BART_SUBSAMPLE and BART_SUBSAMPLE < len(X_tr):
#         idx_b = np.random.RandomState(42).choice(len(X_tr), BART_SUBSAMPLE, replace=False)
#         Xb, yb = X_tr[idx_b], y_tr[idx_b]
#     else:
#         Xb, yb = X_tr, y_tr

#     with localconverter(ro.default_converter + numpy2ri.converter):
#         ro.globalenv['Xb'], ro.globalenv['yb'], ro.globalenv['Xt'] = Xb, yb.astype(float), X_te
#     ro.r('stopifnot(all(sort(unique(yb)) == c(0,1)))')   # garde-fou : probit, pas régression

#     for ntree, k, power in GRID_BART:
#         for nm, val in [('nt', ntree), ('kv', float(k)), ('pw', float(power)),
#                         ('np_', BART_NDPOST), ('ns', BART_NSKIP)]:
#             ro.globalenv[nm] = val
#         t0 = time.time()
#         ro.r('''set.seed(42)
#                 fit  <- bart(x.train=Xb, y.train=yb, x.test=Xt, ntree=nt, k=kv,
#                              power=pw, base=0.95, ndpost=np_, nskip=ns, verbose=FALSE)
#                 p_te <- pnorm(fit$yhat.test); p_tr <- pnorm(fit$yhat.train)''')
#         d_te = np.array(ro.r('p_te'))                     # (ndpost, 71820)
#         MODELS[f"BART t{ntree}_k{k}_pw{power}"] = {
#             'p': np.median(d_te, axis=0), 'draws_ev': d_te[:, m_ev].copy(),
#             'auc_tr': roc_auc_score(yb, np.median(np.array(ro.r('p_tr')), axis=0)),
#             'sec': time.time()-t0}
#         del d_te

# # ================= SORTIES =================
# print(f"\n{'='*112}\n[1] DISCRIMINATION\n{'='*112}")
# auc_tab = pd.DataFrame([{'model': n,
#     'auc_train': v['auc_tr'],
#     'auc_2010':  roc_auc_score(y_cal_bin,  v['p'][m_cal]),
#     'auc_2015':  roc_auc_score(y_true_bin, v['p'][m_ev]),
#     'sec': v['sec']} for n, v in MODELS.items()])
# auc_tab['écart_surapp'] = auc_tab['auc_train'] - auc_tab['auc_2015']
# print(auc_tab.round(4).to_string(index=False))

# print(f"\n{'='*112}\n[2] FRONTIÈRE DE PERTE — seuils calibrés sur 2010, éval 2015\n{'='*112}")
# sweep = pd.DataFrame([{**{'model': n}, **metrics_lambda(v['p'], lam)}
#                       for n, v in MODELS.items() for lam in LAMBDA_GRID])
# for lam in LAMBDA_GRID:
#     print(f"\n--- lambda = {lam} " + "-"*88)
#     print(sweep[sweep['lambda'] == lam].drop(columns='lambda')
#           .sort_values('MAPE').round(4).to_string(index=False))

# print(f"\n{'='*112}\n[3] COUVERTURE PRÉDICTIVE 95% (volume Stan identique partout)\n{'='*112}")
# cov_tab = pd.DataFrame([{'model': n, **coverage(v['p'][m_ev], v['draws_ev'])}
#                         for n, v in MODELS.items()])
# cov_tab['gain_posterior'] = cov_tab['cov_full'] - cov_tab['cov_plugin']
# print(cov_tab.round(4).to_string(index=False))
# print("\ncov_plugin : probabilité ponctuelle, incertitude hurdle IGNORÉE (§4.1).")
# print("cov_full   : tirages du posterior hurdle. NaN pour RF/XGB, qui n'en produisent pas.")
# print("L'écart entre les deux colonnes est la quantité que la §4.1 affirme non nulle.")

# print(f"\n{'='*112}\nCROISEMENT — lambda où le classement bascule\n{'='*112}")
# piv = sweep.pivot_table(index='lambda', columns='model', values=['MAPE','MAE','FN'])
# print(piv.round(1).to_string())


# In[ ]:


#tout ce qui suit travaille sur 2015 uniquement (df_test contient 2010 et 2015)
df_ev   = df_test.loc[m_ev].reset_index(drop=True)
mu_ev   = mu_clean[:, m_ev]
phi_ev  = phi_clean[:, m_ev]
print(f"df_ev : {len(df_ev):,} lignes | mu_ev : {mu_ev.shape}")


# In[ ]:


# #  PRODUCTION FINALE 


#  PRODUCTION FINALE — 2015 uniquement (df_test contient 2010 + 2015)
assert not np.isnan(p_ev).any(), "p_ev contient des NaN"

is_emergent = (df_ev['is_mig_lag'].fillna(0).values == 0)
flow_final  = np.where(is_emergent, flow_q25_ev, flow_med_ev)
y_pred      = np.where(y_pred_bin == 1, flow_final, 0.0)

prob_draws = df_final.filter(like='prob_mig_test').values[valid_draws][:, m_ev]
is_mig_sim = np.random.binomial(1, np.clip(prob_draws, 0, 1))
flow_all   = is_mig_sim * sim_ev
y_pred_q05 = np.percentile(flow_all, 2.5,  axis=0)
y_pred_q95 = np.percentile(flow_all, 97.5, axis=0)

print(f"Ouvertures prédites : {y_pred_bin.sum():,} / {len(y_pred_bin):,} "
      f"(réel : {y_true_bin.sum():,})")


# In[ ]:


acc        = accuracy_score(y_true_bin, y_pred_bin)
global_mae = np.mean(np.abs(y_true - y_pred))
mape_wr    = np.mean(np.abs(y_true - y_pred) / (y_true + 1.0)) * 100
wmape      = np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-8) * 100
log_mae    = np.mean(np.abs(np.log1p(y_true) - np.log1p(y_pred)))
coverage   = np.mean((y_true >= y_pred_q05) & (y_true <= y_pred_q95))

print(f"Accuracy Hurdle   : {acc*100:.1f}%")
print(f"MAE               : {global_mae:,.0f}")
print(f"MAPE (W&R)        : {mape_wr:.1f}%")
print(f"WMAPE             : {wmape:.1f}%")
print(f"Log-MAE           : {log_mae:.4f}")
print(f"Coverage 95%      : {coverage*100:.1f}%")
print()
print(f"{'Modèle':<40} | {'MAE':>10} | {'MAPE':>10} | {'Coverage':>10}")
print("-" * 78)
print(f"{'Welch & Raftery (2022)':<40} | {'~1,200':>10} | {'76.0%':>10} | {'93.0%':>10}")
print(f"{f'Hurdle ARX ZTNB ({N_pays} pays)':<40} | {global_mae:>10,.0f} | {f'{mape_wr:.1f}%':>10} | {f'{coverage*100:.1f}%':>10}")


# In[ ]:


# Tableau de diagnostic bayésien

CLUSTER_LABELS = [SUBREGION_LABELS.get(stan_to_m49.get(k, 99), f'cluster_{k}')
                  for k in range(1, K_clusters + 1)]
Z_LABELS = [f'Z_{k}' for k in range(1, K_Z + 1)]

SCALAIRES = [
    # Volume
    'rho_global_monitor', 'sigma_rho_m49','tau_rho', 'tau_em', 'tau_at', 
    'intercept_em', 'intercept_at', 'phi_disp_global', 'tau_phi_disp',
    # Hurdle
    'mu_beta_lag', 'sigma_beta_lag',
    'intercept_h_em', 'intercept_h_at', 'tau_h_em', 'tau_h_at',
    'tau_u_em',   # échelle du champ spatial : ~0 => l'ICAR ne sert à rien (piste 3)
]

VECTORIELS = {
    'beta_grav'        : X_VOL_COLS,
    'beta_h'           : HURDLE_VARS,      # 12 entrées après le retrait de is_mig_lag
    'beta_lag_m49'     : CLUSTER_LABELS,
    'theta_em'         : Z_LABELS,
    'theta_at'         : Z_LABELS,
    'theta_h_em'       : Z_LABELS,
    'theta_h_at'       : Z_LABELS,
    'phi_disp_cluster' : CLUSTER_LABELS,
    'rho_m49'          : CLUSTER_LABELS,   # moyenne intra-cluster des rho_d (GQ)
}

def ess_bulk(draws):
    from scipy.stats import rankdata
    n = len(draws)
    if n < 4:
        return np.nan
    r = rankdata(draws) / (n + 1)
    z = np.where(r < 0.5, -np.sqrt(2)*np.log(1/(2*r)), np.sqrt(2)*np.log(1/(2*(1-r))))
    if z.var() < 1e-10:
        return n
    ac1 = np.corrcoef(z[:-1], z[1:])[0, 1]
    rho = max(ac1, 0)
    return round(n * (1 - rho) / (1 + rho))

def rhat(chains_draws):
    m = len(chains_draws)
    n = min(len(c) for c in chains_draws)
    chains = np.array([c[:n] for c in chains_draws])
    B = n * np.var(chains.mean(axis=1), ddof=1)
    W = np.mean([np.var(chains[i], ddof=1) for i in range(m)])
    return round(np.sqrt(((n-1)/n * W + B/n) / W), 4) if W > 0 else np.nan

def summarize_param(name, draws_all, chains_draws):
    q = np.percentile(draws_all, [5, 25, 50, 75, 95])
    sig = '*' if (q[0] > 0 or q[4] < 0) else ''
    return {'Paramètre': name, 'Médiane': round(q[2], 4),
            'IC 5%': round(q[0], 4), 'IC 95%': round(q[4], 4),
            'ESS': ess_bulk(draws_all), 'R-hat': rhat(chains_draws), 'Sig': sig}

# découpage par chaîne sur la longueur RÉELLE (df_final = concat des chaînes, dans l'ordre)
n_per_chain = len(df_final) // N_CHAINS
assert n_per_chain * N_CHAINS == len(df_final), "df_final n'est pas un multiple de N_CHAINS"

def _chains(col):
    v = df_final[col].values.astype(float)
    return [v[i*n_per_chain:(i+1)*n_per_chain] for i in range(N_CHAINS)]

rows, manquants = [], []

for param in SCALAIRES:
    if param not in df_final.columns:
        manquants.append(param); continue
    rows.append(summarize_param(param, df_final[param].values.astype(float), _chains(param)))

for param, labels in VECTORIELS.items():
    cols = [c for c in df_final.columns if c.startswith(f'{param}.')]   # notation POINTÉE
    cols = sorted(cols, key=lambda x: int(x.split('.')[1]))
    if not cols:
        manquants.append(param); continue
    if len(cols) != len(labels):
        print(f"[warn] {param} : {len(cols)} colonnes vs {len(labels)} labels")
    for j, col in enumerate(cols):
        label = labels[j] if j < len(labels) else f'{j+1}'
        rows.append(summarize_param(f'{param}[{label}]',
                                    df_final[col].values.astype(float), _chains(col)))

if manquants:
    print(f"[warn] absents de df_final — vérifier vars_to_keep : {manquants}\n")

summary_df = pd.DataFrame(rows)

print(f"{'Paramètre':<35} {'Médiane':>9} {'IC 5%':>9} {'IC 95%':>9} {'ESS':>6} {'R-hat':>7} {'Sig':>4}")
print("-" * 85)
for _, r in summary_df.iterrows():
    flag = ' !' if (r['R-hat'] > 1.01 or r['ESS'] < 400) else ''
    print(f"{r['Paramètre']:<35} {r['Médiane']:>9.4f} {r['IC 5%']:>9.4f} {r['IC 95%']:>9.4f} "
          f"{int(r['ESS']) if not np.isnan(r['ESS']) else 'NaN':>6} {r['R-hat']:>7.4f} {r['Sig']:>4}{flag}")

n_div = int(df_final.get('divergent__', pd.Series([0])).sum())
print(f"\nDivergences : {n_div}")
if 'treedepth__' in df_final.columns:
    print(f"Treedepth saturé (>={MAX_TREEDEPTH}) : {(df_final['treedepth__'] >= MAX_TREEDEPTH).mean()*100:.1f}%")
bad = summary_df[(summary_df['R-hat'] > 1.01) | (summary_df['ESS'] < 400)]
print(f"Paramètres hors seuils : {len(bad)}")


# In[ ]:


# Tableau des coefficients Hurdle et Volume
def print_coef_table(name, means, q05, q95, labels):
    print(f"\n[ {name} ]")
    print(f"{'Variable':<25} {'Moyenne':>10} {'IC 5%':>10} {'IC 95%':>10} {'Sig':>5}")
    print("-" * 65)
    for j in range(len(means)):
        col = labels[j] if j < len(labels) else f'[{j+1}]'
        sig = 'OUI' if (q05[j] > 0 or q95[j] < 0) else 'non'
        print(f"{col:<25} {means[j]:>10.3f} {q05[j]:>10.3f} {q95[j]:>10.3f} {sig:>5}")

# Tableau Hurdle
print_coef_table(
    'HURDLE (Logit)',
    beta_h.mean(axis=0),
    np.percentile(beta_h, 5, axis=0),   # Modifiable à 2.5 si nécessaire pour matcher le IC 95% du graphe
    np.percentile(beta_h, 95, axis=0),  # Modifiable à 97.5
    HURDLE_VARS
)

print_coef_table(
    'VOLUME (ZTNB)',
    beta_grav.mean(axis=0),
    np.percentile(beta_grav, 5, axis=0),
    np.percentile(beta_grav, 95, axis=0),
    X_VOL_COLS
)
rho_m49_draws = df_final.filter(regex=r'^rho_m49\.\d+$').values
phi_cluster_draws   = df_final.filter(like='phi_disp_cluster').values
cluster_labels      = [SUBREGION_LABELS.get(stan_to_m49.get(k, 99), f'cluster_{k}')
                       for k in range(1, K_clusters + 1)]

print_coef_table(
    'RHO AR(1) par cluster M49',
    rho_m49_draws.mean(axis=0),
    np.percentile(rho_m49_draws,  5, axis=0),
    np.percentile(rho_m49_draws, 95, axis=0),
    cluster_labels
)

print_coef_table(
    'PHI dispersion par cluster M49',
    phi_cluster_draws.mean(axis=0),
    np.percentile(phi_cluster_draws,  5, axis=0),
    np.percentile(phi_cluster_draws, 95, axis=0),
    cluster_labels
)

# Figure comparative rho vs phi par cluster
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, draws, title, ylabel in [
    (axes[0], rho_m49_draws,     'Inertie AR(1) par cluster M49',       'rho (entre -1 et 1)'),
    (axes[1], phi_cluster_draws, 'Dispersion NegBin par cluster M49',   'phi_disp_cluster'),
]:
    for k in range(draws.shape[1]):
        ax.violinplot(draws[:, k], positions=[k + 1], widths=0.6, showmedians=True)
    ax.set_xticks(range(1, K_clusters + 1))
    ax.set_xticklabels(cluster_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"rho_phi_cluster_M49_{N_pays}.pdf", bbox_inches='tight')
plt.show()


# In[ ]:


print(rho_m49_draws.shape[1], "colonnes (attendu :", K_clusters, ")")


# In[ ]:


# Figure : hétéroscédasticité M49
fig, ax = plt.subplots(figsize=(14, 5))
for k in range(1, K_clusters + 1):
    draws_k = phi_disp_cluster[:, k-1].flatten()
    ax.violinplot(draws_k, positions=[k], widths=0.6, showmedians=True)
ax.set_xticks(range(1, K_clusters + 1))
ax.set_xticklabels(
    [SUBREGION_LABELS.get(stan_to_m49.get(k, 99), f'Cluster {k}') for k in range(1, K_clusters + 1)],
    rotation=45, ha='right', fontsize=9
)
ax.set_ylabel("phi_disp_cluster (Dispersion inverse)")
ax.set_title(f"Hétéroscédasticité Géographique (M49) — {N_pays} pays\n(phi bas = forte variance)")
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"NegBin_dispersion_cluster_M49_{N_pays}.pdf", bbox_inches='tight')
plt.show()


# In[ ]:


# Figure : coefficients Hurdle et Volume
def plot_coefs(means, q05, q95, labels, title, color_sig, fname):
    K = len(means)
    order = np.argsort(means)
    colors = [color_sig if (q05[i] > 0 or q95[i] < 0) else '#90A4AE' for i in order]
    fig, ax = plt.subplots(figsize=(10, max(5, K * 0.45)))
    ax.barh(
        range(K), means[order],
        xerr=[means[order] - q05[order], q95[order] - means[order]],
        color=colors, alpha=0.85, capsize=3
    )
    ax.set_yticks(range(K))
    ax.set_yticklabels([labels[i] for i in order], fontsize=9)
    ax.axvline(0, color='black', lw=1, ls='--')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

# Figure : Coefficients Hurdle
plot_coefs(
    beta_h.mean(axis=0), 
    np.percentile(beta_h, 2.5, axis=0), 
    np.percentile(beta_h, 97.5, axis=0),
    HURDLE_VARS,
    f"Coefficients Hurdle — {N_pays} pays (IC 95%)\nBleu = IC excluant 0",
    '#2196F3', 
    f"NegBin_hurdle_coefficients_{N_pays}.pdf"
)

plot_coefs(
    beta_grav.mean(axis=0), np.percentile(beta_grav, 2.5, axis=0), np.percentile(beta_grav, 97.5, axis=0),
    X_VOL_COLS,
    f"Coefficients Gravité — {N_pays} pays (IC 90%)\nRouge = IC excluant 0",
    '#F44336', f"NegBin_gravity_coefficients_{N_pays}.pdf"
)


# In[ ]:


# Figure : scatter OOS + distribution des erreurs
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
ax.scatter(y_true, y_pred, alpha=0.4, s=10, color='#1565C0', edgecolors='none')
lim = [0, max(y_true.max(), y_pred.max()) * 1.05]
ax.plot(lim, lim, 'r--', lw=1.5, label='Prédiction parfaite')
ax.set_xscale('symlog')
ax.set_yscale('symlog')
ax.set_xlabel("Flux Réel 2015")
ax.set_ylabel("Flux Prédit")
ax.set_title(f"OOS 2015 — Observé vs Prédit ({N_pays} pays, MAE = {global_mae:,.0f})")
ax.legend()

ax2 = axes[1]
order_err = np.argsort(y_true)
ax2.scatter(range(len(y_true)), np.abs(y_true[order_err] - y_pred[order_err]),
            alpha=0.3, s=8, color='#F44336')
ax2.set_xlabel("Dyades triées par flux réel croissant")
ax2.set_ylabel("|Erreur|")
ax2.set_yscale('log')
ax2.set_title(f"Distribution des erreurs absolues — {N_pays} pays")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"NegBin_prediction_scatter_{N_pays}.pdf", bbox_inches='tight')
plt.show()


# In[ ]:


# Cartographie FP / FN
import plotly.express as px

df_ev['y_true_bin'] = y_true_bin
df_ev['y_pred_bin'] = y_pred_bin
df_ev['FN'] = ((df_ev['y_true_bin'] == 1) & (df_ev['y_pred_bin'] == 0)).astype(int)
df_ev['FP'] = ((df_ev['y_true_bin'] == 0) & (df_ev['y_pred_bin'] == 1)).astype(int)
error_map = df_ev.groupby('orig')[['FN', 'FP']].sum().reset_index()
print(f"FN : {df_ev['FN'].sum()} | FP : {df_ev['FP'].sum()}")


for col, scale, title in [
    ('FN', 'Reds',  'Cartographie Faux Négatifs (FN) par pays d\'origine'),
    ('FP', 'Blues', 'Cartographie Faux Positifs (FP) par pays d\'origine'),
]:
    fig = px.choropleth(
        error_map, locations='orig', color=col,
        hover_name='orig', color_continuous_scale=scale,
        title=title, labels={col: f'Nombre de {col}'}
    )
    fig.update_layout(geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'))
    fig.show()


# # diagnostics

# In[ ]:


y_test = y_true_bin
fp_m = (y_pred_bin==1)&(y_test==0); fn_m=(y_pred_bin==0)&(y_test==1); tp_m=(y_pred_bin==1)&(y_test==1)
n = len(y_test); err = np.abs(y_true - y_pred)
print(f"MAE {err.mean():,.0f}")
print(f"  part FP (volume fantôme) : {y_pred[fp_m].sum()/n:,.1f} ({fp_m.sum()} dyades)")
print(f"  part FN (flux raté)      : {y_true[fn_m].sum()/n:,.1f} ({fn_m.sum()} dyades)")
print(f"  part TP (err. volume)    : {err[tp_m].sum()/n:,.1f} ({tp_m.sum()} dyades)")
print(f"  top-20 dyades = {np.sort(err)[-20:].sum()/err.sum()*100:.0f}% de la MAE totale")


# In[ ]:


err = np.abs(y_true - y_pred)
idx = np.argsort(err)[-20:][::-1]
cols = ['orig','dest','flow','is_mig_lag']
top = df_ev.iloc[idx][cols].assign(pred=y_pred[idx], err=err[idx],
        lag_flow=np.expm1(df_ev['log_flow_lag_clean'].values[idx]))
print(top.to_string())


# In[ ]:


tp_m = (y_pred_bin==1)&(y_test==1)
dec = pd.qcut(y_true[tp_m], 10, labels=False, duplicates='drop')
pd.DataFrame({'dec':dec, 'bias':(y_pred-y_true)[tp_m],
              'log_res':np.log1p(y_pred[tp_m])-np.log1p(y_true[tp_m])}
            ).groupby('dec').agg(['mean','median'])


# In[ ]:


# ============================================================
#  DIAGNOSTIC FAMILLE (c) : décomposition mu_full / rho_d / lag par dyade
#  Sans re-run. rho_raw lu en ciblé dans les CSV (absent de vars_to_keep).
# ============================================================
watch = ['IND_ARE','IND_SAU','IND_OMN','BGD_SAU','BGD_ARE','ARE_IND','BFA_CIV',
         'NPL_IND','CHN_USA']   # 2 sur-prédits en contrôle : leur AR s'engage, lui

# --- draws déjà en RAM (mêmes lignes que mu_clean via valid_draws) ---
alpha_em_d  = df_final.filter(regex=r'^alpha_em\.').values[valid_draws]    # (S, N_pays)
gamma_at_d  = df_final.filter(regex=r'^gamma_at\.').values[valid_draws]
beta_grav_d = df_final.filter(regex=r'^beta_grav\.').values[valid_draws]   # (S, K_v)
rho_lat_d   = df_final.filter(regex=r'^rho_m49_lat\.').values[valid_draws] # (S, K)
tau_rho_d   = df_final['tau_rho'].values[valid_draws]                      # (S,)

# --- rho_raw ciblé : lecture CSV, même ordre de chains que df_final ---
need_dv = {}
for d in watch:
    row = df_volume[df_volume['dyad'] == d]
    if not row.empty:
        need_dv[d] = int(row['dyad_id_v'].iloc[0])          # 1-based Stan
rr_cols = [f'rho_raw.{v}' for v in need_dv.values()]
rr = pd.concat([pd.read_csv(f, comment='#', usecols=rr_cols, engine='c')
                for f in csv_files], ignore_index=True)[rr_cols].values[valid_draws]
rr_j = {v: j for j, v in enumerate(need_dv.values())}

# --- décomposition ---
tt  = df_test.reset_index(drop=True)
key = (tt['orig'] + '_' + tt['dest']).values
X_v_t = np.asarray(X_test_v_std)                            # (N_test, K_v), ordre Stan
clu_v = np.asarray(cluster_v)                               # (D_v,), 1-based

print(f"{'dyade':<9}{'d_v':>7}{'lag01':>6}{'lag_log':>8}{'mu_full':>8}"
      f"{'mu_dt':>8}{'rho_d':>7}{'rho_clu':>8}{'rho_impl':>9}")
for d in watch:
    n = np.where(key == d)[0]
    if len(n) == 0: print(f"{d:<9} absent de df_test"); continue
    n  = int(n[0])
    o  = int(tt['orig_id_test_v'].iloc[n]) - 1
    de = int(tt['dest_id_test_v'].iloc[n]) - 1
    lag01 = float(tt['is_mig_lag'].iloc[n]) if not pd.isna(tt['is_mig_lag'].iloc[n]) else 0.0
    lag = float(tt['log_flow_lag_clean'].iloc[n])
    mu_full_m = float(np.median(alpha_em_d[:, o] + gamma_at_d[:, de] + beta_grav_d @ X_v_t[n]))
    mu_dt_m   = float(np.median(mu_clean[:, n]))
    rho_impl  = (mu_dt_m - mu_full_m) / (lag - mu_full_m) if abs(lag - mu_full_m) > 1e-6 else np.nan
    d_v = need_dv.get(d, 0)
    if d_v > 0:
        k = int(clu_v[d_v - 1]) - 1
        rho_d_m = float(np.median(np.tanh(rho_lat_d[:, k] + tau_rho_d * rr[:, rr_j[d_v]])))
        rho_clu = float(np.tanh(np.median(rho_lat_d[:, k])))
    else:
        rho_d_m, rho_clu = np.nan, np.nan   # d_v=0 -> branche rho_global côté Stan
    print(f"{d:<9}{d_v:>7}{lag01:>6.0f}{lag:>8.2f}{mu_full_m:>8.2f}"
          f"{mu_dt_m:>8.2f}{rho_d_m:>7.3f}{rho_clu:>8.3f}{rho_impl:>9.3f}")


# In[ ]:


key = (df_test['orig'] + '_' + df_test['dest']).values
for d in ['IND_ARE','IND_SAU','BGD_SAU','BFA_CIV','NPL_IND','CHN_USA']:
    n = int(np.where(key == d)[0][0])
    print(f"{d:<9} exp(mu) méd={np.median(np.exp(mu_clean[:, n])):>10,.0f} "
          f"| med_préd={flow_cond_med_final[n]:>10,.0f} "
          f"| phi_d méd={np.median(phi_clean[:, n]):>6.3f} "
          f"| réel={df_test['flow'].values[n]:>10,.0f}")


# In[ ]:


def metrics(yh, tag):
    yh = np.where(y_pred_bin == 1, yh, 0.0)
    mae  = np.abs(y_true - yh).mean()
    mape = (np.abs(y_true - yh) / (y_true + 1)).mean() * 100
    print(f"{tag:<26} MAE = {mae:>8,.0f} | MAPE = {mape:>6.1f}%")

lam_med = np.median(np.exp(np.clip(mu_clean, -50, 50)), axis=0)
p0      = np.exp(-phi_clean * np.log1p(np.exp(np.clip(mu_clean,-50,50) - np.log(phi_clean))))
zt_mean = np.mean(np.exp(np.clip(mu_clean,-50,50)) / np.clip(1 - p0, 1e-6, 1), axis=0)

metrics(flow_cond_med_final,                      "médiane prédictive (actuel)")
metrics(lam_med,                                  "médiane de lambda")
metrics(zt_mean,                                  "moyenne ZT lambda/(1-p0)")
metrics(np.percentile(flow_cond_sim, 75, axis=0), "q75 prédictif")


# In[ ]:


for d in ['IND_ARE','IND_SAU','BGD_SAU','BFA_CIV','NPL_IND','CHN_USA']:
    sub = df_volume[df_volume['dyad'] == d]
    print(f"--- {d}")
    print(sub[['year','flow','is_mig_lag','is_emergent_v','log_flow_lag']].to_string(index=False))


# In[ ]:


from scipy.stats import gamma as _gamma
sub  = np.linspace(0, mu_clean.shape[0]-1, 200).astype(int)   # 200 draws suffisent
lam_s, phi_s = np.exp(np.clip(mu_clean[sub], -50, 50)), phi_clean[sub]
for fl in [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]:
    pe = np.clip(phi_s, max(fl, 1e-8), None)
    yh = np.median(lam_s * _gamma.ppf(0.5, a=pe, scale=1.0/pe), axis=0)
    metrics(yh, f"médiane, phi >= {fl}")


# In[ ]:


fp_m = (y_pred_bin==1)&(y_test==0); fn_m=(y_pred_bin==0)&(y_test==1); tp_m=(y_pred_bin==1)&(y_test==1)
n = len(y_true)

def decomp(yh, tag):
    yhm = np.where(y_pred_bin == 1, yh, 0.0)
    ape = np.abs(y_true - yhm) / (y_true + 1)
    print(f"{tag:<26} MAPE={ape.mean()*100:>6.1f}% "
          f"(FP {ape[fp_m].sum()/n*100:>5.1f} | FN {ape[fn_m].sum()/n*100:>5.1f} | TP {ape[tp_m].sum()/n*100:>5.1f}) "
          f"| MAE={np.abs(y_true-yhm).mean():>8,.0f} "
          f"| part ŷ<1 : {(yhm[y_pred_bin==1] < 1).mean()*100:>4.0f}%")

decomp(flow_cond_med_final, "médiane NB2 (actuel)")
decomp(np.median(lam_s * _gamma.ppf(0.5, a=np.clip(phi_s,1e-8,None), scale=1/np.clip(phi_s,1e-8,None)), axis=0),
       "médiane lambda*nu")
decomp(np.maximum(flow_cond_med_final, 0) * 0 + np.round(np.median(lam_s * _gamma.ppf(0.5, a=np.clip(phi_s,1e-8,None), scale=1/np.clip(phi_s,1e-8,None)), axis=0)),
       "idem, arrondi entier")


# In[ ]:


# --- mu_full sur le TRAIN (médianes posterior) ---
a_m, g_m, b_m = (np.median(alpha_em_d, 0), np.median(gamma_at_d, 0), np.median(beta_grav_d, 0))
Xv = np.asarray(X_vol_std)
mu_full_tr = (a_m[df_volume['orig_id_v'].astype(int).values - 1]
              + g_m[df_volume['dest_id_v'].astype(int).values - 1]
              + Xv @ b_m)

# --- phi_d par dyade, récupéré via les lignes de test (d_v > 0) ---
key_t  = (df_test['orig'] + '_' + df_test['dest']).values
phi_med = np.median(phi_clean, axis=0)
phi_by_dyad = {}
for n in range(len(df_test)):
    if int(df_test['dyad_id_test_v'].iloc[n]) > 0:
        phi_by_dyad.setdefault(key_t[n], phi_med[n])

# --- résidu 1990 vs phi_d ---
m90 = (df_volume['year'].values == df_volume['year'].min())
r90 = np.log(df_volume['flow'].values[m90]) - mu_full_tr[m90]      # écart au niveau structurel
d90 = df_volume['dyad'].values[m90]
ok  = np.array([d in phi_by_dyad for d in d90])
lp  = np.log(np.array([phi_by_dyad[d] for d in d90[ok]]))
print(f"n = {ok.sum():,} | corr(residu_1990, log phi_d) = {np.corrcoef(r90[ok], lp)[0,1]:+.3f}")
for lo, hi in [(-99,1),(1,2),(2,3),(3,4),(4,99)]:
    m = (r90[ok] >= lo) & (r90[ok] < hi)
    if m.sum(): print(f"  residu 1990 in [{lo},{hi}) : n={m.sum():>5,} | phi_d méd = {np.exp(np.median(lp[m])):>7.3f}")


# In[ ]:


from scipy.stats import norm
b_m     = beta_grav[:, X_VOL_COLS.index('flow_momentum')].mean()
rho_lat = df_final.filter(regex=r'^rho_m49_lat\.\d+$').values.mean(axis=0)
tau     = df_final['tau_rho'].mean()
rho_crit = -b_m / (1 - b_m)
lat_crit = np.arctanh(rho_crit)

print(f"beta_momentum = {b_m:+.3f} | rho critique = {rho_crit:.3f}\n")
tot = 0
for k, lat in enumerate(rho_lat):
    frac = norm.cdf((lat_crit - lat) / tau)
    tot += frac
    print(f"  {SUBREGION_LABELS.get(stan_to_m49.get(k+1,99),f'c{k+1}'):<24}"
          f" rho_k={np.tanh(lat):.3f}  P(phi1<0)={frac*100:>5.1f}%")
print(f"\nFraction globale approx. de dyades avec phi1 < 0 : {tot/len(rho_lat)*100:.1f}%")

