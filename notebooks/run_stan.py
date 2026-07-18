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


# Sampling parameters
N_CHAINS        = 4
PARALLEL_CHAINS = 4
ITER_WARMUP     = 500
ITER_SAMPLING   = 400
THIN            = 1
MAX_TREEDEPTH   = 12
ADAPT_DELTA     = 0.95
N_DRAWS         = ITER_SAMPLING // THIN

# Contrôle matériel : vectorized ou multithreading
USE_MULTITHREADING = False  # True (reduce_sum) / False (Vectorisation standard)


# SUBSET DE PAYS (modifier RUN_SIZE uniquement)

RUN_SIZE = 5
# _LABELS  = {1: '50 pays', 2: '80 pays', 3: '110 pays', 4: '140 pays', 5: '190 pays (complet)'}


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


# HURDLE_VARS RF avec colinéarité 
HURDLE_VARS = [
    'log_D_ij', 'log_D_ij_sq', 'COL_ij', 'OL_ij',
    'v2x_polyarchy_o_lag1',# 'v2x_clphy_o_lag1', 'intensity_level_o_lag1',
    'v2x_polyarchy_d_lag1'#, 'v2x_clphy_d_lag1', 'intensity_level_d_lag1'#, 'is_mig_lag'
]

X_VOL_COLS = [
    'log_D_ij', 'log_D_ij_sq', 'LB_ij', 'OL_ij', 'COL_ij', #'t_2000', 't_2000_sq',
    'v2x_polyarchy_o_lag1', 'v2x_clphy_o_lag1', 'intensity_level_o_lag1',
    'v2x_polyarchy_d_lag1', 'v2x_clphy_d_lag1', 'intensity_level_d_lag1'#, 'type_of_conflict_d_lag1',
]

K_grav = len(X_VOL_COLS)
K_h = len(HURDLE_VARS) + 1 # +1 pour logit_xgb

df_train = df[df['year'] <= 2010].copy()
df_test = df[df['year'] == 2015].copy()
df_test_full = df_test.copy()
df_test_full['dyad'] = df_test_full['orig'] + "_" + df_test_full['dest']
df = df_train

HURDLE_REQUIRED = HURDLE_VARS + [ 'is_migration', 'dyad', 'continent_orig',
                                 'is_mig_lag'
                                 ] 
# covariables + is_mig_lag ne devant pas être standardisée et occupant une place théorique particulière (hystérésis) 
# + variables structurelles  dont Stan a besoin pour l'entraînement et la vraisemblance 
# (dyad pour les effets fixes alpha_i et gamma_j, continent_orig pour les effets de cluster M49)
df_hurdle = df.dropna(subset=HURDLE_REQUIRED).copy().reset_index(drop=True)

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

# EXCLUSION PURE des trous pour la ZTNB et le ARX(1)
df_volume = df_volume[is_continu | is_virgin].copy().reset_index(drop=True)

df_volume['is_emergent_v'] = (1 - df_volume['is_mig_lag']).astype(int)
df_volume['log_flow_lag_clean'] = df_volume['log_flow_lag'].fillna(0.0) # Bruit neutralisé par la bifurcation Stan

N_h, N_v = len(df_hurdle), len(df_volume)
print(f"Hurdle : {N_h:,} obs | Volume : {N_v:,} obs sans trous")


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

if 'A2_log' not in HURDLE_VARS:
    HURDLE_VARS = HURDLE_VARS + ['A2_log']
    


# La démonstration clé : A^2 est non-nul là où toutes les variables d'inertie sont muettes
mask_fn_zone = (df_test['is_mig_lag'] == 0) & (df_test.get('log_stock_lag', 0) == 0)
print(f"Zone FN (lag=0, stock=0) : {mask_fn_zone.sum():,} dyades, "
      f"A2>0 pour {(df_test.loc[mask_fn_zone,'A2_log']>0).mean()*100:.1f}% d'entre elles")






from sklearn.ensemble import RandomForestClassifier

RF_VARS = [
    'log_D_ij', 'log_D_ij_sq', 'COL_ij', 'OL_ij',
    #'v2x_polyarchy_o_lag1', 'v2x_clphy_o_lag1', 'intensity_level_o_lag1',
    #'v2x_polyarchy_d_lag1', 'v2x_clphy_d_lag1', 'intensity_level_d_lag1',
    'log_gdpcap_o_lag5', 'log_gdpcap_d_lag5', 'log_gdpcap_diff',
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
    'new_conflict_o', 'new_conflict_d', 'persistent_conflict_o', 'persistent_conflict_d', 'A2_log'
]

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


print(pd.Series(rf_model.feature_importances_, index=RF_VARS_PRESENT).sort_values(ascending=False).head(10).round(4))


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

df_test['dyad'] = df_test['orig'] + "_" + df_test['dest']
df_test['dyad_id_test']   = df_test['dyad'].map(dyad_to_h)
df_test['dyad_id_test_v'] = df_test['dyad'].map(dyad_to_v).fillna(0).astype(int)
df_test = df_test.dropna(subset=['dyad_id_test']).copy().reset_index(drop=True)
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

df_hurdle = df_hurdle.replace([np.inf, -np.inf], np.nan).dropna(subset=HURDLE_REQUIRED)
df_volume = df_volume.replace([np.inf, -np.inf], np.nan).dropna(subset=VOLUME_REQUIRED)
N_h, N_v = len(df_hurdle), len(df_volume)
assert np.isinf(df_volume[X_VOL_COLS].values).sum() == 0

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


audit_paire(df_hurdle, 'intensity_level_d_lag1', 'type_of_conflict_d_lag1')
# audit_paire(df_hurdle, 'A2_log', 'transitivity_proxy')

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

# Test DW spatial (proba RF /!!!\)


# y_true     = df_test['flow'].values
# y_true_bin = (y_true > 0).astype(int)
# prob_med = df_test['proba_rf'].values

# # step1 : résidus du hurdle sur le test OOS 
# # résidu = y_true - y_pred
# # un résidu positif = corridor ouvert sous-estimé ( FN)
# e = y_true_bin.astype(float) - prob_med
# e_c = e - e.mean()  

# # Matrice de contiguïté pays x pays depuis LB_ij du panel
# pays_list = sorted(set(df_test['orig']) | set(df_test['dest']))
# p_idx = {p: i for i, p in enumerate(pays_list)}
# n_pays = len(pays_list)
# C = np.zeros((n_pays, n_pays), dtype=bool)
# lb = df_test[df_test['LB_ij'] == 1][['orig', 'dest']].drop_duplicates()
# for o, d in zip(lb['orig'], lb['dest']):
#     C[p_idx[o], p_idx[d]] = True
#     C[p_idx[d], p_idx[o]] = True   # symétrisation

# def spatial_lag_geo(x, df, C, p_idx, mode='orig'):
#     """Lag géographique : moyenne des résidus des dyades dont l'origine est
#     frontalière de la mienne, à destination identique (mode='orig'),
#     ou symétriquement (mode='dest')."""
#     key_o = df['orig'].map(p_idx).values
#     key_d = df['dest'].map(p_idx).values
#     lag = np.zeros(len(x))
#     # index (origine, destination) -> résidu, via dictionnaire de groupes
#     from collections import defaultdict
#     par_dest = defaultdict(list)   # dest -> liste (orig_idx, position)
#     for pos, (o, d) in enumerate(zip(key_o, key_d)):
#         par_dest[d].append((o, pos))
#     for pos, (o, d) in enumerate(zip(key_o, key_d)):
#         voisins = [p for (o2, p) in par_dest[d] if C[o, o2] and p != pos] if mode=='orig' \
#                   else []
#         lag[pos] = x[voisins].mean() if voisins else 0.0
#     return lag

# lag_geo = spatial_lag_geo(e_c, df_test, C, p_idx, mode='orig')
# DW_geo = (e_c @ lag_geo) / (e_c @ e_c)
# print(f"DW géographique (origines frontalières, même destination) : {DW_geo:.4f}")

# # 
# # Test de CORRELATION SPATIALE sur les résidus du Hurdle
# # H0 : les erreurs de classification sont spatialement indépendantes
# # H1 : les erreurs se regroupent (structure spatiale non captée, il reste de l'info spatiale à capturer)
# # 

# # step2 : définition du voisinage dyadique avec W (non calculée explicitement)
# # deux dyades sont voisines si elles partagent l'origine OU la destination
# # (Mali->Canada est voisine de Mali->France et de Sénégal->Canada)
# orig_codes = df_test['orig'].astype('category').cat.codes.values
# dest_codes = df_test['dest'].astype('category').cat.codes.values
# n = len(e_c)

# def spatial_lag(x, orig_codes, dest_codes):
#     """Moyenne des résidus des voisins (W row-normalisée), sans construire W.
#     Pour chaque dyade : moyenne des résidus partageant l'origine (soi même exclu),
#     idem pour la destination, puis moyenne des deux."""
#     sum_o = np.bincount(orig_codes, weights=x)
#     cnt_o = np.bincount(orig_codes)
#     lag_o = (sum_o[orig_codes] - x) / np.maximum(cnt_o[orig_codes] - 1, 1)

#     sum_d = np.bincount(dest_codes, weights=x)
#     cnt_d = np.bincount(dest_codes)
#     lag_d = (sum_d[dest_codes] - x) / np.maximum(cnt_d[dest_codes] - 1, 1)

#     return 0.5 * (lag_o + lag_d)

# # step3 : statistique de test
# # W étant normalisée par ligne, S0 = n et la formule se réduit à
# # DW = (e' W e) / (e' e), soit une corrélation entre chaque résidu
# # et la moyenne des résidus de ses voisins
# lag_e = spatial_lag(e_c, orig_codes, dest_codes)
# DW_obs = (e_c @ lag_e) / (e_c @ e_c)

# # step4 : p-value (par permutation)
# # on casse la structure spatiale en mélangeant les résidus,
# # et on regarde où se situe le DW observé dans cette distribution nulle
# rng = np.random.default_rng(42)
# n_perm = 999
# DW_perm = np.empty(n_perm)
# for b in range(n_perm):
#     ep = rng.permutation(e_c)
#     DW_perm[b] = (ep @ spatial_lag(ep, orig_codes, dest_codes)) / (ep @ ep)

# p_value = (1 + (DW_perm >= DW_obs).sum()) / (1 + n_perm)
# t_value = (DW_obs - DW_perm.mean()) / DW_perm.std()

# print(f"Stat de test observée : {DW_obs:.4f}")
# print(f"Stat sous H0 : {DW_perm.mean():.4f} (sd {DW_perm.std():.4f})")
# print(f"t-value         : {t_value:.1f}")
# print(f"p-value (perm.) : {p_value:.4f}")

# # décomposition par direction du voisinage 
# # est-ce le partage d'origine ou de destination qui porte la structure ?
# lag_o_only = spatial_lag(e_c, orig_codes, orig_codes * 0)  
# DW_orig = (e_c @ ((np.bincount(orig_codes, weights=e_c)[orig_codes] - e_c)
#                  / np.maximum(np.bincount(orig_codes)[orig_codes] - 1, 1))) / (e_c @ e_c)
# DW_dest = (e_c @ ((np.bincount(dest_codes, weights=e_c)[dest_codes] - e_c)
#                  / np.maximum(np.bincount(dest_codes)[dest_codes] - 1, 1))) / (e_c @ e_c)
# print(f"\nDW côté origine      : {DW_orig:.4f}")
# print(f"DW côté destination  : {DW_dest:.4f}")

# import numpy as np
# from sklearn.metrics import roc_curve

# e = y_true_bin.astype(float) - prob_med

# # Test stat conditionnel aux erreurs fortes > 0.3 (pour éviter la dilution dans le nombre excessif de dyades)
# mask_err = np.abs(e) > 0.3
# e_err = e[mask_err]
# e_c_err = e_err - e_err.mean()

# orig_codes_err = df_test.loc[mask_err, 'orig'].astype('category').cat.codes.values
# dest_codes_err = df_test.loc[mask_err, 'dest'].astype('category').cat.codes.values

# def spatial_lag_err(x, o_codes, d_codes):
#     sum_o = np.bincount(o_codes, weights=x)
#     cnt_o = np.bincount(o_codes)
#     lag_o = (sum_o[o_codes] - x) / np.maximum(cnt_o[o_codes] - 1, 1)

#     sum_d = np.bincount(d_codes, weights=x)
#     cnt_d = np.bincount(d_codes)
#     lag_d = (sum_d[d_codes] - x) / np.maximum(cnt_d[d_codes] - 1, 1)

#     return 0.5 * (lag_o + lag_d)

# DW_obs_err = (e_c_err @ spatial_lag_err(e_c_err, orig_codes_err, dest_codes_err)) / (e_c_err @ e_c_err)

# rng = np.random.default_rng(42)
# n_perm = 999
# DW_perm_err = np.empty(n_perm)
# for b in range(n_perm):
#     ep = rng.permutation(e_c_err)
#     DW_perm_err[b] = (ep @ spatial_lag_err(ep, orig_codes_err, dest_codes_err)) / (ep @ ep)

# p_val_err = (1 + (DW_perm_err >= DW_obs_err).sum()) / (1 + n_perm)
# t_value_err = (DW_obs_err - DW_perm_err.mean()) / DW_perm_err.std()

# DW_orig_err = (e_c_err @ ((np.bincount(orig_codes_err, weights=e_c_err)[orig_codes_err] - e_c_err)
#                          / np.maximum(np.bincount(orig_codes_err)[orig_codes_err] - 1, 1))) / (e_c_err @ e_c_err)
# DW_dest_err = (e_c_err @ ((np.bincount(dest_codes_err, weights=e_c_err)[dest_codes_err] - e_c_err)
#                          / np.maximum(np.bincount(dest_codes_err)[dest_codes_err] - 1, 1))) / (e_c_err @ e_c_err)

# print("second refinement: CONDITIONNEL aux fortes erreurs (|e| > 0.3)")
# print(f"Stat de test observée  : {DW_obs_err:.4f}")
# print(f"t-value       : {t_value_err:.1f}")
# print(f"p-value      : {p_val_err:.4f}")
# print(f"DW côté origine (neighbourhood = share origin)  : {DW_orig_err:.4f}")
# print(f"DW côté dest (neighbourhood = share dest)     : {DW_dest_err:.4f}\n")


# # third refinement : Voisinage Régional M49 (Mêmes blocs m49 d'origine)
# e_c_global = e - e.mean()
# m49_orig_codes = df_test['continent_orig'].astype('category').cat.codes.values
# dest_codes_global = df_test['dest'].astype('category').cat.codes.values

# def spatial_lag_m49(x, m49_codes, d_codes):
#     sum_m49 = np.bincount(m49_codes, weights=x)
#     cnt_m49 = np.bincount(m49_codes)
#     lag_m49 = (sum_m49[m49_codes] - x) / np.maximum(cnt_m49[m49_codes] - 1, 1)

#     sum_d = np.bincount(d_codes, weights=x)
#     cnt_d = np.bincount(d_codes)
#     lag_d = (sum_d[d_codes] - x) / np.maximum(cnt_d[d_codes] - 1, 1)

#     return 0.5 * (lag_m49 + lag_d)

# DW_obs_m49 = (e_c_global @ spatial_lag_m49(e_c_global, m49_orig_codes, dest_codes_global)) / (e_c_global @ e_c_global)

# DW_perm_m49 = np.empty(n_perm)
# for b in range(n_perm):
#     ep = rng.permutation(e_c_global)
#     DW_perm_m49[b] = (ep @ spatial_lag_m49(ep, m49_orig_codes, dest_codes_global)) / (ep @ ep)

# p_val_m49 = (1 + (DW_perm_m49 >= DW_obs_m49).sum()) / (1 + n_perm)
# t_value_m49 = (DW_obs_m49 - DW_perm_m49.mean()) / DW_perm_m49.std()

# DW_orig_m49 = (e_c_global @ ((np.bincount(m49_orig_codes, weights=e_c_global)[m49_orig_codes] - e_c_global)
#                             / np.maximum(np.bincount(m49_orig_codes)[m49_orig_codes] - 1, 1))) / (e_c_global @ e_c_global)

# print("third refinement : regional structure M49 ")
# print(f"Stat de test observée (neighbourhood = M49)     : {DW_obs_m49:.4f}")
# print(f"t-value (M49)             : {t_value_m49:.1f}")
# print(f"p-value (M49)             : {p_val_m49:.4f}")
# print(f"DW côté cluster M49 orig  : {DW_orig_m49:.4f}")





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


# import cmdstanpy
# cmdstanpy.rebuild_cmdstan()   # long : plusieurs minutes
