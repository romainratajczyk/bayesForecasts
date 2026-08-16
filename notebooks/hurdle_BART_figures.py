#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[3]:


# Publication figure style.

#
# The pgf backend hands every string in the figure to pdflatex, so the maths in
# an axis label is set in the same glyphs as the maths in the article. The cost
# is that inline previews stop working. Set BAYESMIG_TEX=0 before this cell to
# fall back on matplotlib's Computer Modern fonts, which is much faster while
# iterating and keeps the preview.
import sys
from pathlib import Path

sys.path.insert(0, ".")
import figstyle as fs

fs.use_style()

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# Display names. The article is in English and its equations use these symbols,
# so the figures have to as well.
PRETTY = {
    "log_D_ij": r"$\log D_{ij}$",
    "log_D_ij_sq": r"$(\log D_{ij})^{2}$",
    "LB_ij": "Shared border",
    "OL_ij": "Common language",
    "COL_ij": "Colonial tie",
    "v2x_polyarchy_o_lag5": r"Polyarchy, origin $(t{-}1)$",
    "v2x_polyarchy_d_lag5": r"Polyarchy, destination $(t{-}1)$",
    "v2x_clphy_o_lag5": r"Physical integrity, origin $(t{-}1)$",
    "v2x_clphy_d_lag5": r"Physical integrity, destination $(t{-}1)$",
    "intensity_level_o_lag5": r"Conflict intensity, origin $(t{-}1)$",
    "intensity_level_d_lag5": r"Conflict intensity, destination $(t{-}1)$",
    "flow_momentum": r"Flow momentum $\Delta_{ij,t-1}$",
    "A2_log": r"$\log(1+A^{2}_{ij})$",
}

SUBREGION_EN = {
    11: "Northern Europe", 12: "Southern Europe", 13: "Western Europe",
    14: "Eastern Europe", 15: "Northern Africa", 16: "Western Africa",
    17: "Eastern Africa", 18: "Middle Africa", 19: "Southern Africa",
    21: "Northern America", 22: "Central America", 23: "Caribbean",
    24: "South America", 30: "Eastern Asia", 34: "Southern Asia",
    35: "South-eastern Asia", 53: "Oceania", 143: "Central Asia",
    145: "Western Asia", 99: "Unclassified",
}


def pretty(name):
    return PRETTY.get(name, name.replace("_", " "))


def cluster_labels_en(stan_to_m49, k_clusters):
    return [SUBREGION_EN.get(stan_to_m49.get(k, 99), f"cluster {k}")
            for k in range(1, k_clusters + 1)]


# In[4]:


# Sampling parameters
N_CHAINS        = 4
PARALLEL_CHAINS = 4
ITER_WARMUP     = 1000
ITER_SAMPLING   = 1200
THIN            = 1
MAX_TREEDEPTH   = 12
ADAPT_DELTA     = 0.95
N_DRAWS         = ITER_SAMPLING // THIN

# Contrôle matériel : vectorized ou multithreading
USE_MULTITHREADING = False  # True (reduce_sum) / False (Vectorisation standard)


# SUBSET DE PAYS (modifier RUN_SIZE uniquement)

RUN_SIZE = 5
# _LABELS  = {1: '50 pays', 2: '80 pays', 3: '110 pays', 4: '140 pays', 5: '190 pays (complet)'}


# In[5]:


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


# In[6]:


# 1. Base 50 : Échantillon fondamental. Diversité géographique maximale et intégration des plus grands pôles.
PAYS_SUBSET_50 = {
    # Amérique du Nord & Sud
    'USA', 'CAN', 'MEX', 'BRA', 'ARG', 'COL', 'CHL', 'PER', 'VEN',
    # Europe
    'FRA', 'DEU', 'GBR', 'ITA', 'ESP', 'POL', 'RUS', 'UKR', 'SWE', 'NLD', 'ROU',
    # Asie & Moyen-Orient
    'CHN', 'IND', 'JPN', 'KOR', 'IDN', 'PAK', 'BGD', 'PHL', 'VNM', 'TUR', 'IRN', 'SAU', 'THA', 'MYS', 'KAZ',
    # Afrique
    #'NGA', 'ETH', 'EGY', 'COD', 'ZAF', 'TZA', 'KEN', 'DZA', 'MAR', 'GHA', 'CIV', 'AGO', 'SEN',
    # Océanie
    #'AUS', 'NZL'
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


# In[7]:


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


# In[8]:


df['dyad'] = df['orig'] + "_" + df['dest']

# df['instability_o'] = df['v2x_clphy_o_lag1'] - df['v2x_polyarchy_o_lag1']
# df['instability_d'] = df['v2x_clphy_d_lag1'] - df['v2x_polyarchy_d_lag1']

df = df.sort_values(['orig', 'dest', 'year']).reset_index(drop=True)


df = df.dropna(subset=['is_mig_lag']).reset_index(drop=True)

GRAVITY_VARS_RAW = ['P_it', 'P_jt', 'PSR_i', 'PSR_j', 'IMR_it', 'IMR_jt', 'urban_it', 'urban_jt', 'LA_i', 'LA_j']
for raw in GRAVITY_VARS_RAW:
    df[f'log_{raw}'] = np.log(df[raw].replace(0, np.nan)) # créer les variables log


# In[9]:


# === Momentum du flux : Δ = log Y_{t-1} − log Y_{t-2} (pente, absente de l'ARX(1)) ===
df = df.sort_values(['dyad', 'year'])
lag1_raw = df.groupby('dyad')['log_flow'].shift(1)         # brut (NaN si fermé), PAS le patch
lag2_raw = df.groupby('dyad')['log_flow'].shift(2)
df['flow_momentum'] = (lag1_raw - lag2_raw).fillna(0.0)    # NaN -> 0 : pas de tendance observée
df = df.sort_values(['orig', 'dest', 'year']).reset_index(drop=True)
print(f"flow_momentum non-nul : {(df['flow_momentum'] != 0).mean()*100:.1f}% | "
      f"range [{df['flow_momentum'].min():.2f}, {df['flow_momentum'].max():.2f}]")


# In[ ]:


# HURDLE_VARS RF avec colinéarité 
HURDLE_VARS = [
    'log_D_ij', 'log_D_ij_sq', 'COL_ij', 'OL_ij',
    'v2x_polyarchy_o_lag5',# 'v2x_clphy_o_lag1', 'intensity_level_o_lag1',
    'v2x_polyarchy_d_lag5'#, 'v2x_clphy_d_lag1', 'intensity_level_d_lag1'#, 'is_mig_lag'
]

X_VOL_COLS = [
    'log_D_ij', 'LB_ij', 'OL_ij', 'COL_ij', #'t_2000', 't_2000_sq', 'log_D_ij_sq',
    'v2x_polyarchy_o_lag5', 'intensity_level_o_lag5', #'v2x_clphy_o_lag5'
     #'v2x_clphy_d_lag5', 'intensity_level_d_lag5'#, 'type_of_conflict_d_lag1', 'v2x_polyarchy_d_lag1'
]




df_train = df[df['year'] <= 2010].copy()
#df_test = df[df['year'] == 2015].copy()
df_hold   = df[df['year'] == 2010].copy()      # calibration OOS
df_2015   = df[df['year'] == 2015].copy()
df_test   = pd.concat([df_hold, df_2015], ignore_index=True)
df_test['is_2015'] = (df_test['year'] == 2015).astype(int)
df_test_full = df_test.copy()
df_test_full['dyad'] = df_test_full['orig'] + "_" + df_test_full['dest']
df = df_train


# In[11]:


HURDLE_REQUIRED = HURDLE_VARS + [ 'is_migration', 'dyad', 'continent_orig',
                                 'is_mig_lag'
                                 ] 
# covariables + is_mig_lag ne devant pas être standardisée et occupant une place théorique particulière (hystérésis) 
# + variables structurelles  dont Stan a besoin pour l'entraînement et la vraisemblance 
# (dyad pour les effets fixes alpha_i et gamma_j, continent_orig pour les effets de cluster M49)

# 1995-2005 uniquement pour  exclusion de 1990 (biais is_mig_lag) et vraie calibration OOS des FN/FP


df_hurdle = df[(df['year'] >= 1995) & (df['year'] <= 2005)] \
    .dropna(subset=HURDLE_REQUIRED) \
    .copy() \
    .reset_index(drop=True)

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


# volume: 69k trop léger?
# hurdle: 140k bien assez, on peut enlever 1990? 

# In[12]:


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
#X_h_std,   stats_h   = standardize_matrix(df_hurdle[HURDLE_VARS].values, HURDLE_VARS, BINARY_COLS_HUR)


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
#X_test_h_std, _ = standardize_matrix(df_test[HURDLE_VARS].values, HURDLE_VARS, BINARY_COLS_HUR, fit_stats=stats_h)

df_test['cluster_test'] = df_test['continent_orig_fill'].values   # renommage

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


# In[16]:


df_hurdle = df_hurdle.replace([np.inf, -np.inf], np.nan).dropna(subset=HURDLE_REQUIRED)
df_volume = df_volume.replace([np.inf, -np.inf], np.nan).dropna(subset=VOLUME_REQUIRED)
N_h, N_v = len(df_hurdle), len(df_volume)
assert np.isinf(df_volume[X_VOL_COLS].values).sum() == 0

_sc = df_volume.groupby('dyad')['log_flow_lag_clean'].mean().reindex(dyades_v).values
stan_data_extra_scale = (_sc - _sc.mean()) / max(_sc.std(), 1e-8)

stan_data = {
    'N_pays'   : N_pays_total,
    'K_Z'      : int(K_Z),
    'Z_em'     : Z_em.tolist(),
    'Z_at'     : Z_at.tolist(),
    'K_clusters': int(K_clusters),

    'N_v'      : int(N_v),
    'D_v'      : int(D_v),
    'K_v'      : int(K_grav),
    'dyad_id_v': df_volume['dyad_id_v'].astype(int).tolist(),
    'orig_id_v': df_volume['orig_id_v'].astype(int).tolist(),
    'dest_id_v': df_volume['dest_id_v'].astype(int).tolist(),
    'flow'         : df_volume['flow'].astype(int).tolist(),
    'log_flow_lag' : df_volume['log_flow_lag_clean'].astype(float).tolist(),
    'momentum_v'   : df_volume['flow_momentum'].astype(float).tolist(),
    'is_emergent_v': df_volume['is_emergent_v'].astype(int).tolist(),
    'X_v'          : X_vol_std.tolist(),
    'log_scale_v'  : stan_data_extra_scale.tolist(),
    'cluster_v'    : cluster_v.tolist(),

    'N_test'           : int(len(df_test)),
    'dyad_id_test_v'   : df_test['dyad_id_test_v'].astype(int).tolist(),
    'orig_id_test_v'   : df_test['orig_id_test_v'].astype(int).tolist(),
    'dest_id_test_v'   : df_test['dest_id_test_v'].astype(int).tolist(),
    'X_v_test'         : X_test_v_std.tolist(),
    'log_flow_lag_test': df_test['log_flow_lag_clean'].tolist(),
    'momentum_test'    : df_test['flow_momentum'].astype(float).tolist(),
    'is_mig_lag_test'  : df_test['is_mig_lag'].fillna(0.0).tolist(),
    'cluster_test'     : df_test['cluster_test'].astype(int).tolist(),

    'do_loo': 0,
}

assert len(stan_data['is_mig_lag_test']) == stan_data['N_test']
assert len(stan_data['cluster_test'])    == stan_data['N_test']
assert min(stan_data['dyad_id_test_v'])  >= 0 and max(stan_data['cluster_test']) <= K_clusters


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


# HURDLE BART — remplace intégralement le logit hiérarchique
# PLACEMENT OBLIGATOIRE : après la construction de stan_data,
# donc après les dropna de df_hurdle et df_test.
# Indépendant de Stan (vraisemblance séparable) : peut tourner
# avant, après ou en parallèle de l'échantillonnage HMC.

# import numpy as np, time
# import rpy2.robjects as ro
# from rpy2.robjects import numpy2ri
# from rpy2.robjects.conversion import localconverter
# from sklearn.metrics import roc_auc_score

# #  PARAMÈTRES 
# NTREE, K_BART, POWER, BASE = 200, 2.0, 2.0, 0.95   # défauts Chipman et al. (2010)
# NDPOST, NSKIP = 1000, 500
# SUBSAMPLE = None          # None = 140 624 lignes (~6 min)
# SEED_BART = 42


# # Jeu de covariables du hurdle (ex-RF_VARS, renommé : plus de forêt aléatoire)
# BART_VARS = [
#     'log_D_ij', 'log_D_ij_sq', 'OL_ij', 'COL_ij', 'LB_ij',
#     'log_gdpcap_o_lag5', 'log_gdpcap_d_lag5',
#     'log_P_it', 'log_P_jt',
#     'PSR_i', 'PSR_j', 'IMR_it', 'IMR_jt', 'urban_it', 'urban_jt',
#     'LL_i', 'LL_j', 'LA_i', 'LA_j',
#     'v2x_polyarchy_o_lag5', 'v2x_polyarchy_d_lag5',
#     'is_mig_lag', 'log_stock_lag',
#     'transitivity_proxy', 'A2_log', 'flow_momentum',
# ]

# manquantes_tr = [c for c in BART_VARS if c not in df_hurdle.columns]
# manquantes_te = [c for c in BART_VARS if c not in df_test.columns]
# assert not manquantes_tr, f"absentes de df_hurdle : {manquantes_tr}"
# assert not manquantes_te, f"absentes de df_test : {manquantes_te}"

# X_tr = df_hurdle[BART_VARS].fillna(0).values          # même ordre de colonnes
# y_tr = df_hurdle['is_migration'].values.astype(float)
# X_te = df_test[BART_VARS].fillna(0).values
# assert X_tr.shape[1] == X_te.shape[1] == len(BART_VARS)
# assert X_te.shape[0] == len(df_test) == stan_data['N_test'], \
#     "df_test figé ? la cellule BART doit venir APRÈS le dropna et stan_data"

# print(f"BART | {len(BART_VARS)} covariables")
# print(f"  train {X_tr.shape} années {sorted(df_hurdle['year'].unique())}")
# print(f"  test  {X_te.shape} (2010 + 2015)")

# y_ref = y_tr
# if SUBSAMPLE and SUBSAMPLE < len(X_tr):
#     sub = np.random.RandomState(SEED_BART).choice(len(X_tr), SUBSAMPLE, replace=False)
#     X_tr, y_tr = X_tr[sub], y_tr[sub]
#     y_ref = y_tr

# ro.r('library(dbarts)')
# with localconverter(ro.default_converter + numpy2ri.converter):
#     ro.globalenv['Xb'], ro.globalenv['yb'], ro.globalenv['Xt'] = X_tr, y_tr, X_te
# ro.r('stopifnot(all(sort(unique(yb)) == c(0,1)))')   # garde-fou : probit, pas régression

# for nm, val in [('nt', NTREE), ('kv', float(K_BART)), ('pw', float(POWER)),
#                 ('bs', float(BASE)), ('np_', NDPOST), ('ns', NSKIP),
#                 ('sd', SEED_BART)]:
#     ro.globalenv[nm] = val

# t0 = time.time()
# ro.r('''set.seed(sd)
#         fit <- bart(x.train = Xb, y.train = yb, x.test = Xt,
#                     ntree = nt, k = kv, power = pw, base = bs,
#                     ndpost = np_, nskip = ns, verbose = FALSE)
#         p_te <- pnorm(fit$yhat.test)
#         p_tr <- pnorm(fit$yhat.train)''')

# p_bart_draws = np.asarray(ro.r('p_te'), dtype=np.float32)      # (NDPOST, N_test)
# p_bart_train = np.median(np.asarray(ro.r('p_tr')), axis=0)
# ro.r('rm(fit); gc()')

# assert p_bart_draws.shape[1] == len(df_test), "désalignement p_bart_draws / df_test"
# df_test['p_hurdle'] = np.median(p_bart_draws, axis=0)

# m_cal_ = (df_test['is_2015'] == 0).values
# m_ev_  = (df_test['is_2015'] == 1).values
# print(f"\nBART : {time.time()-t0:.0f}s | draws {p_bart_draws.shape} "
#       f"| {p_bart_draws.nbytes/1e6:.0f} Mo")
# print(f"AUC  train {roc_auc_score(y_ref, p_bart_train):.4f}"
#       f" | 2010 {roc_auc_score((df_test.loc[m_cal_,'flow']>0).astype(int), df_test.loc[m_cal_,'p_hurdle']):.4f}"
#       f" | 2015 {roc_auc_score((df_test.loc[m_ev_,'flow']>0).astype(int), df_test.loc[m_ev_,'p_hurdle']):.4f}")

# # Part des corridors passant par le relais kappa plutôt que par l'ARX (article §4.2)
# part_kappa = (df_test.loc[m_ev_, 'is_mig_lag'].fillna(0).values < 0.5).mean()
# print(f"Bifurcation OOS 2015 : {part_kappa*100:.1f}% des dyades via kappa")


# In[18]:


print("dyad_id_test_h : min =", df_test['dyad_id_test'].min(),
      "| hors-train =", int((df_test['dyad'].map(dyad_to_h).isna()).sum()))
print("dyad_id_test_v : min =", df_test['dyad_id_test_v'].min(),
      "| d_v=0 =", int((df_test['dyad_id_test_v'] == 0).sum()))
print("N_test =", len(df_test), "| dont 2010 :", int((df_test['year'] == 2010).sum()),
      "| dont 2015 :", int((df_test['year'] == 2015).sum()))


# In[19]:


assert min(stan_data['dyad_id_test_v']) >= 0
assert len(stan_data['X_v_test']) == stan_data['N_test'] == len(df_test)
assert len(stan_data['log_scale_v']) == D_v
assert stan_data['K_v'] == len(X_VOL_COLS)
print(f"K_v = {stan_data['K_v']} | OK, prêt pour Stan")


# In[20]:


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
    STAN_FILE = "../STAN/HMC_BART_vectorized.stan"



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


# In[21]:


CSV_PREFIX    = f"ARX_{N_pays}pays_{N_CHAINS}c_{ITER_SAMPLING}it_{arch_suffix}"
csv_files = [
    f"{OUTPUT_DIR}/{CSV_PREFIX}_chain{i+1}.csv"
    for i in range(N_CHAINS)
]

csv_files = ['/Users/rratajczyk/Desktop/bayesForecasts/bayesTemp/onyxia data/stan_outputs_tmux/stan_outputs_tmux/stan_outputs_tmux/stan_outputs_tmux/ARX_190pays_4c_403it_VECT_chain1.csv',
             '/Users/rratajczyk/Desktop/bayesForecasts/bayesTemp/onyxia data/stan_outputs_tmux/stan_outputs_tmux/stan_outputs_tmux/stan_outputs_tmux/ARX_190pays_4c_403it_VECT_chain2.csv',
             '/Users/rratajczyk/Desktop/bayesForecasts/bayesTemp/onyxia data/stan_outputs_tmux/stan_outputs_tmux/stan_outputs_tmux/stan_outputs_tmux/ARX_190pays_4c_403it_VECT_chain3.csv',
             '/Users/rratajczyk/Desktop/bayesForecasts/bayesTemp/onyxia data/stan_outputs_tmux/stan_outputs_tmux/stan_outputs_tmux/stan_outputs_tmux/ARX_190pays_4c_403it_VECT_chain4.csv']
with open(csv_files[0], 'r') as f:
    for line in f:
        if not line.startswith('#'):
            all_cols = line.strip().split(',')
            break

vars_to_keep = [
    'alpha_em', 'gamma_at', 'beta_grav',
    'intercept_em', 'intercept_at', 'theta_em', 'theta_at', 'tau_em', 'tau_at',
    'rho_global', 'rho_m49', 'rho_m49_lat', 'sigma_rho_m49', 'tau_rho',
    'kappa_m49', 'mu_kappa', 'sigma_kappa', 'omega',
    'phi_disp_global', 'phi_disp_cluster', 'tau_phi_disp', 'delta_phi',
    'mu_dt_test', 'phi_test',
    'lp__', 'divergent__', 'treedepth__', 'energy__', 'stepsize__',
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


# In[22]:


mu_test = df_final.filter(like='mu_dt_test').values
phi_t   = df_final.filter(like='phi_test').values
beta_grav_draws  = df_final.filter(like='beta_grav').values
phi_disp_cluster = df_final.filter(like='phi_disp_cluster').values
rho_m49_draws    = df_final.filter(regex=r'^rho_m49\.\d+$').values
kappa_draws      = df_final.filter(regex=r'^kappa_m49\.\d+$').values




print(f"mu_test shape : {mu_test.shape}")


# In[23]:


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


# In[24]:


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
# prob_draws = df_final.filter(like='prob_mig_test').values[valid_draws][:, m_ev]
rng   = np.random.RandomState(0)
S_vol = flow_cond_sim.shape[0]
idx_b = rng.choice(p_bart_draws.shape[0], S_vol, replace=(S_vol > p_bart_draws.shape[0]))
prob_draws = p_bart_draws[idx_b][:, m_ev]

is_mig_sim = rng.binomial(1, np.clip(prob_draws, 0, 1))
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


# In[25]:


for tag, mk in [('tous', np.ones(len(y_true), bool)),
                ('flux > 0', y_true_bin == 1), ('flux = 0', y_true_bin == 0)]:
    cv = np.mean((y_true[mk] >= y_pred_q05[mk]) & (y_true[mk] <= y_pred_q95[mk]))
    lg = np.mean(y_pred_q95[mk] - y_pred_q05[mk])
    print(f"{tag:<10} n={mk.sum():>6,}  couverture {cv*100:5.1f}%  largeur moy. {lg:>10,.0f}")


# In[26]:


#tout ce qui suit travaille sur 2015 uniquement (df_test contient 2010 et 2015)
df_ev   = df_test.loc[m_ev].reset_index(drop=True)
mu_ev   = mu_clean[:, m_ev]
phi_ev  = phi_clean[:, m_ev]
print(f"df_ev : {len(df_ev):,} lignes | mu_ev : {mu_ev.shape}")


# In[27]:


# #  PRODUCTION FINALE 


#  PRODUCTION FINALE — 2015 uniquement (df_test contient 2010 + 2015)
assert not np.isnan(p_ev).any(), "p_ev contient des NaN"

is_emergent = (df_ev['is_mig_lag'].fillna(0).values == 0)
flow_final  = np.where(is_emergent, flow_q25_ev, flow_med_ev)
y_pred      = np.where(y_pred_bin == 1, flow_final, 0.0)
# prob_draws = df_final.filter(like='prob_mig_test').values[valid_draws][:, m_ev]
rng   = np.random.RandomState(0)
S_vol = flow_cond_sim.shape[0]
idx_b = rng.choice(p_bart_draws.shape[0], S_vol, replace=(S_vol > p_bart_draws.shape[0]))
prob_draws = p_bart_draws[idx_b][:, m_ev]
is_mig_sim = rng.binomial(1, np.clip(prob_draws, 0, 1))

print(f"Ouvertures prédites : {y_pred_bin.sum():,} / {len(y_pred_bin):,} "
      f"(réel : {y_true_bin.sum():,})")


# In[28]:


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


# In[29]:


# Tableau de diagnostic bayésien

CLUSTER_LABELS = [SUBREGION_LABELS.get(stan_to_m49.get(k, 99), f'cluster_{k}')
                  for k in range(1, K_clusters + 1)]
Z_LABELS = [f'Z_{k}' for k in range(1, K_Z + 1)]

SCALAIRES = [
    'rho_global', 'sigma_rho_m49', 'tau_rho', 'tau_em', 'tau_at',
    'intercept_em', 'intercept_at', 'phi_disp_global', 'tau_phi_disp',
    'mu_kappa', 'sigma_kappa', 'omega', 'delta_phi',
]
VECTORIELS = {
    'beta_grav'        : X_VOL_COLS,
    'theta_em'         : Z_LABELS,
    'theta_at'         : Z_LABELS,
    'phi_disp_cluster' : CLUSTER_LABELS,
    'rho_m49'          : CLUSTER_LABELS,
    'kappa_m49'        : CLUSTER_LABELS,
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


# In[30]:


# Tableau des coefficients Hurdle et Volume
def print_coef_table(name, means, q05, q95, labels):
    print(f"\n[ {name} ]")
    print(f"{'Variable':<25} {'Moyenne':>10} {'IC 5%':>10} {'IC 95%':>10} {'Sig':>5}")
    print("-" * 65)
    for j in range(len(means)):
        col = labels[j] if j < len(labels) else f'[{j+1}]'
        sig = 'OUI' if (q05[j] > 0 or q95[j] < 0) else 'non'
        print(f"{col:<25} {means[j]:>10.3f} {q05[j]:>10.3f} {q95[j]:>10.3f} {sig:>5}")

# beta_grav n'existe pas dans ce notebook : la cellule qui extrait les draws
# definit beta_grav_draws.
beta_grav = beta_grav_draws

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

# Figure : inertie AR(1) et dispersion par sous-region M49.
# Half-eye plutot que violon : le violon est symetrique et gaspille donc la
# moitie de son encre, et il ne montre pas l'intervalle credible. Les deux
# panneaux partagent l'ordre du premier, pour que chaque sous-region occupe la
# meme ligne des deux cotes.
labels_en = cluster_labels_en(stan_to_m49, K_clusters)

fig, axes = plt.subplots(1, 2, figsize=(fs.TEXT_WIDTH, 0.30 * K_clusters + 1.0),
                         sharey=True, layout="constrained")
order = fs.halfeye(axes[0], rho_m49_draws, labels=labels_en, sort_by="median",
                   color=fs.BLUE)
axes[0].set_xlabel(r"AR(1) inertia $\rho_k$")
fs.halfeye(axes[1], phi_cluster_draws, labels=labels_en, sort_by=order,
           color=fs.VERMILION)
axes[1].set_xlabel(r"Dispersion $\phi_k$ (lower is more dispersed)")
axes[1].tick_params(axis="y", labelleft=False)
fs.save(fig, "fig_subregion_heterogeneity", FIG_DIR)


# In[31]:


print(rho_m49_draws.shape[1], "colonnes (attendu :", K_clusters, ")")


# In[32]:


# Figure : heteroscedasticite par sous-region M49.
# Pas de titre : ce que la figure montre se dit dans le \caption{} du .tex.
labels_en = cluster_labels_en(stan_to_m49, K_clusters)

fig, ax = fs.figure(height=0.30 * K_clusters + 1.0)
fs.halfeye(ax, phi_disp_cluster, labels=labels_en, sort_by="median",
           color=fs.VERMILION)
ax.set_xlabel(r"Negative binomial dispersion $\phi_k$ (lower is more dispersed)")
fs.save(fig, "fig_dispersion_by_subregion", FIG_DIR)


# In[33]:


# Figure : coefficients du modele de volume.
# Dot-whisker et non barh : un coefficient est une position, pas une longueur
# partant de zero. Deux intervalles emboites, 50% epais et 95% fin, pour que la
# borne exterieure ne se lise pas comme une frontiere nette. Plus de codage
# couleur "l'intervalle exclut zero" : cela reintroduit le test de
# significativite frequentiste dans une lecture bayesienne.
fig, ax = fs.figure(height=0.30 * len(X_VOL_COLS) + 0.85)
fs.interval_plot(ax, beta_grav, labels=[pretty(c) for c in X_VOL_COLS])
ax.set_xlabel("Posterior coefficient, standardised covariates")
fs.save(fig, "fig_gravity_coefficients", FIG_DIR)


# In[34]:


# Figure : previsions hors echantillon 2015.
# Panneau gauche : densite hexbin plutot qu'un nuage a alpha=0.4, ou 36 000
# points se superposent et masquent la ou la masse se trouve reellement. Le
# hexbin est rasterise, le texte reste vectoriel.
# Panneau droit : concentration de l'erreur, qui dit ce que la distribution des
# erreurs absolues triees essayait de dire, en lisible.
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

ax = axes[1]
abs_err = np.abs(y_true - y_pred)
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


# In[35]:


# Cartographie FP / FN par pays d'origine.
# La choropleth plotly reste utile pour explorer, mais elle sort en raster et
# une carte de comptages est biaisee par la surface des pays. Pour l'article,
# un classement par pays est vectoriel et se lit sans ambiguite.


df_ev['y_true_bin'] = y_true_bin
df_ev['y_pred_bin'] = y_pred_bin
df_ev['FN'] = ((df_ev['y_true_bin'] == 1) & (df_ev['y_pred_bin'] == 0)).astype(int)
df_ev['FP'] = ((df_ev['y_true_bin'] == 0) & (df_ev['y_pred_bin'] == 1)).astype(int)
error_map = df_ev.groupby('orig')[['FN', 'FP']].sum().reset_index()
print(f"FN : {df_ev['FN'].sum()} | FP : {df_ev['FP'].sum()}")

by_origin = (error_map.set_index('orig')
             .assign(total=lambda d: d['FN'] + d['FP'])
             .sort_values('total', ascending=False)
             .head(22)
             .sort_values('total'))

fig, ax = fs.figure(height=0.26 * len(by_origin) + 0.9)
rows = np.arange(len(by_origin))
fs.row_bands(ax, len(by_origin))
ax.barh(rows, -by_origin['FN'], height=0.62, color=fs.VERMILION,
        edgecolor="none", zorder=2)
ax.barh(rows, by_origin['FP'], height=0.62, color=fs.BLUE,
        edgecolor="none", zorder=2)
ax.axvline(0, color=fs.INK, lw=0.6, zorder=3)
ax.set_yticks(rows)
ax.set_yticklabels(by_origin.index)
ax.set_ylim(-0.6, len(by_origin) - 0.4)
# Deux etiquettes, chacune dans la couleur des barres qu'elle nomme, plutot
# qu'une seule que le lecteur devrait desambiguiser.
ax.text(0.25, -0.055, "Missed openings", transform=ax.transAxes, ha="center",
        va="top", color=fs.VERMILION, fontsize=8)
ax.text(0.75, -0.055, "Spurious openings", transform=ax.transAxes, ha="center",
        va="top", color=fs.BLUE, fontsize=8)
fs.despine(ax, left=True)
ax.xaxis.grid(True, color=fs.RULE, lw=0.4)
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _p: f"{abs(v):,.0f}".replace(",", r"\," if fs._USE_TEX else " ")))
fs.save(fig, "fig_error_by_origin", FIG_DIR)

