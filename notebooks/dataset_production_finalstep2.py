import pandas as pd
import numpy as np
import country_converter as coco
import logging

logging.getLogger('country_converter').setLevel(logging.ERROR)


# CHEMINS


INPUT_R    = "../data/panel_june_R.csv"
PATH_VDEM  = "../data/V-Dem-CY-Core-v16.csv"
PATH_UCDP  = "../data/UcdpPrioConflict_v25_1.csv"
PATH_STOCK = "../data/undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx"
OUTPUT     = "../data/panel_june_ready.csv"

YEARS = [1990, 1995, 2000, 2005, 2010, 2015]

# CHARGEMENT DE LA SORTIE R (étape 1 de le création du dataset final)

df = pd.read_csv(INPUT_R)
df = df[df['orig'] != df['dest']].copy()
print(f"Base R chargée : {df.shape[0]:,} lignes, {df['orig'].nunique()} pays")

# Exclusions nécessaires (ajouter manuellement les 4 PIB manquants cependant)

PAYS_EXCLURE = {
    'SSD',  # Indépendance juillet 2011
    'MNE',  # Indépendance juin 2006
    'TLS',  # Indépendance 2002
    'CUW',  # Autonomie octobre 2010
    'GUM',  # Territoire américain
    'MYT',  # Territoire français
    'VIR',  # Territoire américain
    'CLI',  # Territoire australien
}

# données de PIB à intégrer: MYT VIR CLI GUM au moins à partir de 2000-2010

df = df[
    ~df['orig'].isin(PAYS_EXCLURE) &
    ~df['dest'].isin(PAYS_EXCLURE)
].copy()

print(f"Après exclusions Run B : {df['orig'].nunique()} pays")
for pays in ['PSE', 'COD', 'ROU', 'SRB']:
    present = pays in df['orig'].unique()
    n_obs   = (df['orig'] == pays).sum()
    print(f"  {pays} : {'présent' if present else 'ABSENT'} — {n_obs} obs comme origine")

df = df.sort_values(['orig', 'dest', 'year']).reset_index(drop=True)


#  VARIABLES DÉRIVÉES 


df['dyad']          = df['orig'] + "_" + df['dest']
df['is_migration']  = (df['flow'] > 0).astype(int)
df['log_flow']      = np.where(df['flow'] > 0, np.log(df['flow']), np.nan)
df['log_flow_lag']  = df.groupby('dyad')['log_flow'].shift(1)
df['is_mig_lag']    = df.groupby('dyad')['is_migration'].shift(1)

df['log_D_ij']      = np.log(df['D_ij'].replace(0, np.nan))
df['log_D_ij_sq']   = df['log_D_ij'] ** 2
df['logD_times_LB'] = df['log_D_ij'] * df['LB_ij']

SEUIL_LOG_GDP   = 2.9
df['is_rich_o'] = (df['log_gdpcap_o_lag'] > SEUIL_LOG_GDP).astype(float)

df['log_gdpcap_diff'] = df['log_gdpcap_d_lag'] - df['log_gdpcap_o_lag']

df['t_2000']    = df['year'] - 2000
df['t_2000_sq'] = (df['year'] - 2000) ** 2

# Patchs autoreg

# t=1990 : première période, pas de lag — imputation à 0 (prior d'inactivité)
mask_1990 = (df['year'] == 1990) & df['is_mig_lag'].isna()
df.loc[mask_1990, 'is_mig_lag'] = 0.0
print(f"Patch is_mig_lag t=1990 : {mask_1990.sum():,} obs récupérées")

# Ouvertures de couloirs : flow > 0 mais log_flow_lag NaN → imputation à 0
mask_ouverture = df['log_flow_lag'].isna() & (df['flow'] > 0)
df.loc[mask_ouverture, 'log_flow_lag'] = 0.0
print(f"Patch log_flow_lag ouvertures : {mask_ouverture.sum():,} obs récupérées")

# Seul dropna légitime au niveau dataset
n_avant = len(df)
df = df.dropna(subset=['is_mig_lag']).reset_index(drop=True)
print(f"Dropna is_mig_lag : -{n_avant - len(df):,} obs (première période résiduelle)")


# VARIABLES GÉOPOLITIQUES V-DEM (lag1 et lag5)

df_vdem = pd.read_csv(
    PATH_VDEM,
    usecols=['country_text_id', 'year', 'v2x_polyarchy', 'v2x_clphy']
)
df_vdem = df_vdem.rename(columns={'country_text_id': 'iso3'})

# lag1
df_vdem_lag1 = df_vdem.copy()
df_vdem_lag1['year_merge'] = df_vdem_lag1['year'] + 1
df_vdem_lag1 = df_vdem_lag1.drop(columns=['year'])

# lag5
df_vdem_lag5 = df_vdem.copy()
df_vdem_lag5['year_merge'] = df_vdem_lag5['year'] + 5
df_vdem_lag5 = df_vdem_lag5.drop(columns=['year'])

for side in ['orig', 'dest']:
    suffix = '_o' if side == 'orig' else '_d'

    df = df.merge(
        df_vdem_lag1.rename(columns={'iso3': side, 'year_merge': 'year'}),
        on=[side, 'year'], how='left'
    ).rename(columns={
        'v2x_polyarchy': f'v2x_polyarchy{suffix}_lag1',
        'v2x_clphy':     f'v2x_clphy{suffix}_lag1',
    })

    df = df.merge(
        df_vdem_lag5.rename(columns={'iso3': side, 'year_merge': 'year'}),
        on=[side, 'year'], how='left'
    ).rename(columns={
        'v2x_polyarchy': f'v2x_polyarchy{suffix}_lag5',
        'v2x_clphy':     f'v2x_clphy{suffix}_lag5',
    })

    # bfill/ffill intra-pays puis médiane globale pour les micro-états
    for var in ['v2x_polyarchy', 'v2x_clphy']:
        for lag in ['lag1', 'lag5']:
            col = f'{var}{suffix}_{lag}'
            df[col] = df.groupby(side)[col].transform(lambda x: x.bfill().ffill())
            df[col] = df[col].fillna(df[col].median())

# 
# VARIABLES CONFLICTUELLES UCDP (lag1 et lag5 + fenêtres glissantes)


cc = coco.CountryConverter()

df_ucdp_raw = pd.read_csv(
    PATH_UCDP,
    usecols=['gwno_loc', 'year', 'intensity_level', 'type_of_conflict']
, sep=None,engine='python')
df_ucdp_raw['gwno_loc'] = df_ucdp_raw['gwno_loc'].astype(str).str.split(',')
df_ucdp_raw = df_ucdp_raw.explode('gwno_loc')
df_ucdp_raw['gwno_loc'] = pd.to_numeric(df_ucdp_raw['gwno_loc'].str.strip(), errors='coerce')
df_ucdp_raw = df_ucdp_raw.dropna(subset=['gwno_loc'])
df_ucdp_raw['iso3'] = cc.convert(
    names=df_ucdp_raw['gwno_loc'].tolist(), src='GWcode', to='ISO3', not_found=np.nan
)
df_ucdp_raw = df_ucdp_raw.dropna(subset=['iso3'])

df_ucdp_annual = df_ucdp_raw.groupby(['iso3', 'year'])[
    ['intensity_level', 'type_of_conflict']
].max().reset_index()

# Agrégations lag1 et lag5
df_ucdp_lag1 = df_ucdp_annual.copy()
df_ucdp_lag1['year_merge'] = df_ucdp_lag1['year'] + 1
df_ucdp_lag1 = df_ucdp_lag1.drop(columns=['year'])

df_ucdp_lag5 = df_ucdp_annual.copy()
df_ucdp_lag5['year_merge'] = df_ucdp_lag5['year'] + 5
df_ucdp_lag5 = df_ucdp_lag5.drop(columns=['year'])

for side in ['orig', 'dest']:
    suffix = '_o' if side == 'orig' else '_d'

    df = df.merge(
        df_ucdp_lag1.rename(columns={'iso3': side, 'year_merge': 'year'}),
        on=[side, 'year'], how='left'
    ).rename(columns={
        'intensity_level':   f'intensity_level{suffix}_lag1',
        'type_of_conflict':  f'type_of_conflict{suffix}_lag1',
    })
    df[f'intensity_level{suffix}_lag1'].fillna(0, inplace=True)
    df[f'type_of_conflict{suffix}_lag1'].fillna(0, inplace=True)

    df = df.merge(
        df_ucdp_lag5.rename(columns={'iso3': side, 'year_merge': 'year'}),
        on=[side, 'year'], how='left'
    ).rename(columns={
        'intensity_level':   f'intensity_level{suffix}_lag5',
        'type_of_conflict':  f'type_of_conflict{suffix}_lag5',
    })
    df[f'intensity_level{suffix}_lag5'].fillna(0, inplace=True)
    df[f'type_of_conflict{suffix}_lag5'].fillna(0, inplace=True)

# Fenêtres glissantes [t-5, t-1]
conflict_window = []
for year_cible in YEARS + [2020]:
    window = df_ucdp_annual[
        (df_ucdp_annual['year'] >= year_cible - 5) &
        (df_ucdp_annual['year'] <= year_cible - 1)
    ].groupby('iso3').agg(
        any_conflict_window=('intensity_level',  lambda x: int((x > 0).any())),
        max_conflict_window=('intensity_level',  'max'),
        any_intense_window= ('intensity_level',  lambda x: int((x >= 2).any())),
        any_intl_window=    ('type_of_conflict', lambda x: int((x >= 3).any())),
    ).reset_index()
    window['year'] = year_cible
    conflict_window.append(window)

df_conflict_window = pd.concat(conflict_window, ignore_index=True)

for side in ['orig', 'dest']:
    suffix = '_o' if side == 'orig' else '_d'
    df = df.merge(
        df_conflict_window.rename(columns={
            'iso3':               side,
            'any_conflict_window': f'any_conflict{suffix}_window',
            'max_conflict_window': f'max_conflict{suffix}_window',
            'any_intense_window':  f'any_intense{suffix}_window',
            'any_intl_window':     f'any_intl{suffix}_window',
        }),
        on=[side, 'year'], how='left'
    )

window_cols = [
    'any_conflict_o_window', 'max_conflict_o_window',
    'any_intense_o_window',  'any_intl_o_window',
    'any_conflict_d_window', 'max_conflict_d_window',
    'any_intense_d_window',  'any_intl_d_window',
]
df[window_cols] = df[window_cols].fillna(0)

# new_conflict et persistent_conflict
df['new_conflict_o'] = (
    (df['intensity_level_o_lag1'] > 0) & (df['intensity_level_o_lag5'] == 0)
).astype(int)
df['new_conflict_d'] = (
    (df['intensity_level_d_lag1'] > 0) & (df['intensity_level_d_lag5'] == 0)
).astype(int)
df['persistent_conflict_o'] = (
    (df['intensity_level_o_lag1'] > 0) & (df['intensity_level_o_lag5'] > 0)
).astype(int)
df['persistent_conflict_d'] = (
    (df['intensity_level_d_lag1'] > 0) & (df['intensity_level_d_lag5'] > 0)
).astype(int)

# Index composite instabilité (cohérent avec la spec du modèle)
df['instability_o'] = df['v2x_clphy_o_lag1'] - df['v2x_polyarchy_o_lag1']
df['instability_d'] = df['v2x_clphy_d_lag1'] - df['v2x_polyarchy_d_lag1']

# 
#  STOCKS UN DESA (proxy diaspora, lag +5)


df_stock = pd.read_excel(PATH_STOCK, sheet_name="Table 1", header=10)

df_stock['orig'] = cc.convert(
    names=df_stock['Location code of origin'].tolist(),
    src='UN', to='ISO3', not_found=np.nan
)
df_stock['dest'] = cc.convert(
    names=df_stock['Location code of destination'].tolist(),
    src='UN', to='ISO3', not_found=np.nan
)
df_stock = df_stock.dropna(subset=['orig', 'dest'])

annees_stock = [c for c in df_stock.columns
                if (isinstance(c, int) and c in YEARS)
                or (isinstance(c, str) and c.isdigit() and int(c) in YEARS)]

df_stock_long = df_stock.melt(
    id_vars=['orig', 'dest'],
    value_vars=annees_stock,
    var_name='year_stock',
    value_name='stock_migrants'
)
df_stock_long['year_stock']    = df_stock_long['year_stock'].astype(int)
df_stock_long['stock_migrants'] = pd.to_numeric(
    df_stock_long['stock_migrants'].replace('..', 0), errors='coerce'
).fillna(0)

# Le stock de t est le proxy de diaspora pour t+5
df_stock_long['year']          = df_stock_long['year_stock'] + 5
df_stock_long['log_stock_lag'] = np.log1p(df_stock_long['stock_migrants'])

df = df.merge(
    df_stock_long[['orig', 'dest', 'year', 'log_stock_lag']],
    on=['orig', 'dest', 'year'], how='left'
)
df['log_stock_lag'] = df['log_stock_lag'].fillna(0)


# PERSISTANCE CONFLICTUELLE (rolling sur 2 périodes quinquennales)


df = df.sort_values(['orig', 'dest', 'year']).reset_index(drop=True)

for col_base, group_key in [
    ('intensity_level_o_lag1',   'orig'),
    ('type_of_conflict_o_lag1',  'orig'),
    ('intensity_level_d_lag1',   'dest'),
    ('type_of_conflict_d_lag1',  'dest'),
]:
    other = 'dest' if group_key == 'orig' else 'orig'
    new_col = f'{col_base}_persist'
    df[new_col] = (
        df.groupby([group_key, other])[col_base]
        .transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean())
    )

# NETTOYAGE DES COLONNES MORTES


# Colonnes produites par le R mais inutiles au modèle
# Les niveaux bruts GDP sont redondants avec les versions log et per-capita
COLS_TO_DROP = [
    'ihs_flow', 'log_flow_plus_1', 'flow_raw',
    'gdp_o',      'gdp_d',
    'gdp_o_lag1', 'gdp_d_lag1',
    'gdp_o_lag5', 'gdp_d_lag5',
    'gdpcap_o_lag1', 'gdpcap_d_lag1',
    'gdpcap_o_lag5', 'gdpcap_d_lag5',
    'gdpcap_o',   'gdpcap_d',
    'log_gdpcap_o', 'log_gdpcap_d',
    'log_gdp_o',  'log_gdp_d',
    'log_gdp_o_lag', 'log_gdp_d_lag',
    'log_gdp_o_lag1', 'log_gdp_d_lag1',
    'log_gdp_o_lag5', 'log_gdp_d_lag5',
    'gdpcap_o_lag',   'gdpcap_d_lag',   # alias R, remplacés par log_gdpcap_o/d_lag
    'iso3_o', 'iso3_d',                 # doublons de orig/dest
    'cod_o', 'cod_d',                   # codes numériques ONU intermédiaires
]
df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], inplace=True)

# Arrondi flottant (empreinte mémoire)
float_cols = df.select_dtypes(include=['float64', 'float32']).columns
df[float_cols] = df[float_cols].round(5)

# Audit final et EXPORT

print(f"\nDimensions finales : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
print(f"Pays : {df['orig'].nunique()}")
print(f"Années : {sorted(df['year'].unique().tolist())}")

# Taux de NaN sur les variables clés du modèle
VARS_CLES = [
    'log_D_ij', 'log_D_ij_sq', 'LB_ij', 'OL_ij', 'COL_ij',
    'log_gdpcap_o_lag', 'log_gdpcap_d_lag', 'log_gdpcap_diff',
    'v2x_polyarchy_o_lag1', 'v2x_clphy_o_lag1', 'intensity_level_o_lag1',
    'v2x_polyarchy_d_lag1', 'v2x_clphy_d_lag1', 'intensity_level_d_lag1',
    'is_mig_lag', 'log_flow_lag', 'log_stock_lag',
]
print("\nTaux de NaN sur les variables clés :")
for col in VARS_CLES:
    if col in df.columns:
        pct = df[col].isna().mean() * 100
        flag = "  !" if pct > 5 else ""
        print(f"  {col:<35} {pct:5.1f}%{flag}")
    else:
        print(f"  {col:<35} ABSENTE")

# Vérification topologie (Δt = 5 ans strictement)
df_check = df.copy()
df_check['delta_t'] = df_check.groupby('dyad')['year'].diff()
anomalies = df_check[~df_check['delta_t'].isin([5.0, np.nan])]
if anomalies.empty:
    print("\nTopologie validée : panel strictement continu (Δt = 5 ans)")
else:
    print(f"\nRupture topologique : {len(anomalies)} observations asynchrones")
    print(anomalies[['orig', 'dest', 'year', 'delta_t']].head())

df.to_csv(OUTPUT, index=False)
print(f"\nDataset exporté : {OUTPUT}")
