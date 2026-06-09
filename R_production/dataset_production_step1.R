# CODE ETAPE 1 CREATION DU DATASET FINAL

library(data.table)
library(readxl)
library(wpp2019)
library(countrycode)
library(WDI)
library(magrittr)

options(warn = 0)
# Option sans dépendance - à insérer avant la boucle
`%||%` <- function(l, r) if (is.null(l)) r else l

# REMAPPING ISO3 (problème avec ROU ROM)

# Raison : les anciennes bases utilisent des codes obsolètes. Si on harmonise
# après les jointures, les lignes ne matchent jamais.

ISO_REMAP <- c(
  "ROM" = "ROU",   # Roumanie : ancien code FAOSTAT/CEPII
  "ZAR" = "COD",   # RD Congo : ancien code pré-1997
  "SCG" = "SRB",   # Serbie-Monténégro : dissolution 2006
  # MNE (Monténégro) apparaît parfois sous SCG également — on duplique
  "YUG" = "SRB",   # Yougoslavie résiduelle dans certaines bases
  "WBG" = "PSE",   # Palestine : World Bank Group code
  "PSE" = "PSE",   # déjà correct mais parfois "oPt" dans certaines bases
  "TMP" = "TLS",   # Timor-Leste : ancien code
  "SDN" = "SDN"    # Soudan : resté SDN, SSD est le Sud-Soudan post-2011
)

harmonize_iso3 <- function(dt, cols) {
  # Applique le remapping sur les colonnes ISO3 spécifiées
  for (col in cols) {
    if (col %in% names(dt)) {
      dt[, (col) := toupper(get(col))]
      for (old_code in names(ISO_REMAP)) {
        dt[get(col) == old_code, (col) := ISO_REMAP[old_code]]
      }
    }
  }
  return(dt)
}

# Chargement et harmonisation
#getwd()=='/Users/romain/Desktop/Projets DS/ProjetStat'
top200     <- fread("/Users/romain/Desktop/Projets DS/ProjetStat/data/200isoRegionCodes.csv")
flows      <- fread("/Users/romain/Desktop/Projets DS/ProjetStat/data/abelCohen2019flowsv6_flowdt.csv")
dist_cepii <- as.data.table(read_excel("/Users/romain/Desktop/Projets DS/ProjetStat/data/dist_cepii.xls"))
geo_cepii  <- as.data.table(read_excel("/Users/romain/Desktop/Projets DS/ProjetStat/data/geo_cepii.xls"))

# Harmonisation AVANT toute jointure
top200     <- harmonize_iso3(top200,     c("iso"))
flows      <- harmonize_iso3(flows,      c("orig", "dest"))
dist_cepii <- harmonize_iso3(dist_cepii, c("iso_o", "iso_d"))
geo_cepii  <- harmonize_iso3(geo_cepii,  c("iso3"))


#  TABLE DISTANCES COMPLETE
# Pour les pays absents de dist_cepii (CUW, GUM, MYT, VIR, TLS, SSD, PSE...),
# on calcule la distance via les coordonnées capitales (données en dur, inattaquables).
# Source : CIA World Factbook / Wikipedia — coordonnées des capitales


# Coordonnées capitales pour les pays manquants dans dist_cepii
CAPITALS <- data.table(
  iso3 = c("CUW", "GUM", "MYT", "VIR", "TLS", "SSD", "PSE", "MNE", "SRB", "COD","CLI"),
  lat  = c(12.11, 13.47, -12.78, 18.34,  -8.56,  4.85, 31.90, 42.44, 44.80, -4.32,10.30),
  lon  = c(-68.93, 144.79, 45.23, -64.90, 125.57, 31.57, 35.20, 19.26, 20.46, 15.32,-109.21)
)

haversine_dt <- function(lat1, lon1, lat2, lon2) {
  R <- 6371
  phi1 <- lat1 * pi / 180; phi2 <- lat2 * pi / 180
  dphi <- (lat2 - lat1) * pi / 180
  dlam <- (lon2 - lon1) * pi / 180
  a <- sin(dphi/2)^2 + cos(phi1) * cos(phi2) * sin(dlam/2)^2
  2 * R * asin(sqrt(a))
}

# Extraire les coordonnées de tous les pays déjà dans dist_cepii
# dist_cepii ne contient pas de lat/lon : on crée une table de ref
# à partir des paires connues (on utilise geo_cepii si lat/lon présents)
# Sinon fallback : calculer uniquement les distances manquantes

# Construire dist_clean depuis dist_cepii (agrégation par paire)
dist_clean <- dist_cepii[, .(
  D_ij    = mean(distcap, na.rm = TRUE),
  LB_ij   = max(contig,       na.rm = TRUE),
  OL_ij   = max(comlang_off,  na.rm = TRUE),
  COL_ij  = max(colony,       na.rm = TRUE)
), by = .(iso_o = iso_o, iso_d = iso_d)]

# Identifier les pays manquants dans dist_clean
pays_presents <- union(dist_clean$iso_o, dist_clean$iso_d)
pays_cibles   <- top200$iso
pays_manquants <- setdiff(pays_cibles, pays_presents)
cat("Pays manquants dans dist_cepii après harmonisation :", pays_manquants, "\n")

# Pour les pays manquants, créer toutes les paires avec les pays présents
# en calculant la distance haversine depuis les capitales
if (length(pays_manquants) > 0) {
  # Coordonnées de référence pour les pays DÉJÀ dans dist_cepii
  # On récupère les coordonnées depuis geo_cepii si disponibles
  # Sinon on utilise uniquement les capitales hardcodées ci-dessus

  # Vérifier si geo_cepii a des lat/lon
  geo_has_coords <- all(c("lat", "lon") %in% names(geo_cepii))
  cat("geo_cepii contient lat/lon :", geo_has_coords, "\n")

  if (geo_has_coords) {
    ref_coords <- geo_cepii[iso3 %in% pays_cibles, .(iso3, lat, lon)]
  } else {
    # Fallback : uniquement les capitales hardcodées
    ref_coords <- CAPITALS[iso3 %in% pays_cibles]
    cat("AVERTISSEMENT : coordonnées limitées aux", nrow(ref_coords),
        "pays hardcodés. Distances manquantes pour les autres.\n")
  }

  caps_manquantes <- CAPITALS[iso3 %in% pays_manquants, .(iso3, lat, lon)]
  ref_coords <- rbindlist(list(ref_coords, caps_manquantes), fill = TRUE)
  ref_coords <- unique(ref_coords, by = "iso3")

  # Créer les lignes manquantes
  lignes_nouvelles <- list()
  for (pays_m in pays_manquants) {
    coord_m <- CAPITALS[iso3 == pays_m]
    if (nrow(coord_m) == 0) {
      cat("  Pas de coordonnées pour", pays_m, "— paires ignorées\n")
      next
    }
    # Création des tables avec clé factice pour produit cartésien
    dt_o <- data.table(iso_o = pays_m, lat_o = coord_m$lat, lon_o = coord_m$lon, dummy = 1L)
    dt_d <- ref_coords[, .(iso_d = iso3, lat_d = lat, lon_d = lon, dummy = 1L)]

    # Jointure sur la clé factice et suppression de celle-ci
    paires <- merge(dt_o, dt_d, by = "dummy", allow.cartesian = TRUE)
    paires[, dummy := NULL]
    
    paires[, D_ij := haversine_dt(lat_o, lon_o, lat_d, lon_d)]

    # LB_ij, OL_ij, COL_ij : imputation par règles géographiques
    # Valeur par défaut : 0 (pas de frontière, pas de langue commune, pas de colonie)
    # Les exceptions connues sont listées explicitement
    CONTIG_KNOWN <- list(
      "SSD" = c("CAF", "COD", "ETH", "KEN", "SDN", "UGA"),
      "MNE" = c("ALB", "BIH", "HRV", "SRB"),
      "TLS" = c("IDN"),
      "PSE" = c("ISR", "JOR", "EGY")
    )
    OL_KNOWN <- list(
      "PSE" = c("JOR", "EGY", "SAU", "SYR", "LBN", "IRQ", "YEM", "ARE",
                "KWT", "QAT", "OMN", "BHR", "DZA", "TUN", "LBY", "MAR"),
      "SSD" = c("ETH", "SDN"),
      "MNE" = c("SRB", "HRV", "BIH"),  # serbo-croate
      "TLS" = c("PRT")                   # portugais
    )
    COL_KNOWN <- list(
      "PSE" = c("GBR"),
      "SSD" = c("GBR"),
      "TLS" = c("PRT", "IDN"),
      "CUW" = c("NLD"),
      "VIR" = c("USA"),
      "GUM" = c("USA")
    )

    paires[, LB_ij  := as.integer(iso_d %in% (CONTIG_KNOWN[[pays_m]] %||% character(0)))]
    paires[, OL_ij  := as.integer(iso_d %in% (OL_KNOWN[[pays_m]]  %||% character(0)))]
    paires[, COL_ij := as.integer(iso_d %in% (COL_KNOWN[[pays_m]] %||% character(0)))]

    nouvelles_o <- paires[, .(iso_o, iso_d, D_ij, LB_ij, OL_ij, COL_ij)]
    nouvelles_d <- paires[, .(iso_o = iso_d, iso_d = iso_o, D_ij,
                               LB_ij, OL_ij, COL_ij)]  # symétrie
    lignes_nouvelles <- c(lignes_nouvelles, list(nouvelles_o, nouvelles_d))
  }

  if (length(lignes_nouvelles) > 0) {
    dist_extra  <- rbindlist(lignes_nouvelles, fill = TRUE)
    dist_clean  <- rbindlist(list(dist_clean, dist_extra), fill = TRUE)
    dist_clean  <- unique(dist_clean, by = c("iso_o", "iso_d"))
    cat("Lignes ajoutées à dist_clean :", nrow(dist_extra), "\n")
  }
}


# DONNÉES PAYS 


data(UNlocations)
iso_map <- as.data.table(UNlocations)[location_type == 4, .(country_code, name)]
iso_map[, iso3 := toupper(suppressWarnings(countrycode(country_code, "iso3n", "iso3c")))]
iso_map <- harmonize_iso3(iso_map, "iso3")
iso_map <- unique(iso_map[!is.na(iso3) & iso3 %in% top200$iso])

data(popM); data(popF); data(mxM); data(mxF)
years_vec <- seq(1990, 2015, 5)
years_str <- as.character(years_vec)

m_dt <- melt(as.data.table(popM), id.vars = c("country_code", "age"),
             measure.vars = years_str, variable.name = "year", value.name = "m")
f_dt <- melt(as.data.table(popF), id.vars = c("country_code", "age"),
             measure.vars = years_str, variable.name = "year", value.name = "f")
m_dt[, year := as.numeric(as.character(year))]
f_dt[, year := as.numeric(as.character(year))]

country_stats <- merge(m_dt, f_dt, by = c("country_code", "age", "year"))[
  , .(tot = m + f), by = .(country_code, age, year)]
country_stats <- country_stats[, .(
  P_t = sum(tot),
  psr = sum(tot[age %in% c("15-19","20-24","25-29","30-34","35-39",
                            "40-44","45-49","50-54","55-59","60-64")]) /
        sum(tot[age %in% c("65-69","70-74","75-79","80-84","85-89",
                            "90-94","95-99","100+")])
), by = .(country_code, year)]

mx_cols <- grep("1990|1995|2000|2005|2010|2015", names(mxM), value = TRUE)
imr_dt <- merge(
  melt(as.data.table(mxM)[age == 0], id.vars = "country_code",
       measure.vars = mx_cols, value.name = "imr_m"),
  melt(as.data.table(mxF)[age == 0], id.vars = "country_code",
       measure.vars = mx_cols, value.name = "imr_f"),
  by = c("country_code", "variable")
)
imr_dt[, `:=`(year   = as.numeric(substr(variable, 1, 4)),
               IMR_t  = (imr_m + imr_f) / 2)]
country_stats <- merge(country_stats, imr_dt[, .(country_code, year, IMR_t)],
                       by = c("country_code", "year"))

# PIB : on récupère t-5 ET t-1 pour permettre les deux specs dans Python
# t-5 lag  : year + 5 pour que la valeur de 1990 matche avec 1995 (lag quinquennal)
# t-1 lag  : year + 1 pour que la valeur de 1989 matche avec 1990
wdi_raw <- as.data.table(WDI(
  indicator = c("urban" = "SP.URB.TOTL.IN.ZS", "gdp" = "NY.GDP.MKTP.CD"),
  start = 1984, end = 2015, extra = FALSE   # étendu à 1984 pour le lag t-5 de 1990
))
setnames(wdi_raw, old = c("iso3c"), new = c("iso3"), skip_absent = TRUE)
wdi_raw <- harmonize_iso3(wdi_raw, "iso3")
wdi_raw <- wdi_raw[!is.na(iso3) & iso3 %in% top200$iso]

# Lag t-1 (GDP de l'année précédente)
gdp_lag1 <- wdi_raw[, .(iso3, year = year + 1, PIB_lag1 = gdp)]

# Lag t-5 (GDP quinquennal — correspond à la spécification originale)
gdp_lag5 <- wdi_raw[, .(iso3, year = year + 5, PIB_lag5 = gdp)]

wdi_curr <- wdi_raw[year %in% years_vec, .(iso3, year, urban_t = urban, PIB = gdp)]
wdi_final <- merge(wdi_curr, gdp_lag1, by = c("iso3", "year"), all.x = TRUE)
wdi_final <- merge(wdi_final, gdp_lag5, by = c("iso3", "year"), all.x = TRUE)

geo_clean <- geo_cepii[, .(
  LA = mean(area,       na.rm = TRUE),
  LL = max(landlocked,  na.rm = TRUE)
), by = .(iso3 = iso3)]

country_stats <- merge(country_stats, iso_map[, .(country_code, iso3)],
                       by = "country_code")
country_stats <- merge(country_stats, wdi_final,  by = c("iso3", "year"), all.x = TRUE)
country_stats <- merge(country_stats, geo_clean,  by = "iso3",            all.x = TRUE)

# Assemblage

master_dt <- flows[orig %in% top200$iso & dest %in% top200$iso]
master_dt <- merge(master_dt, iso_map[, .(iso3, country_code)],
                   by.x = "orig", by.y = "iso3", all.x = TRUE)
setnames(master_dt, "country_code", "cod_o")
master_dt <- merge(master_dt, iso_map[, .(iso3, country_code)],
                   by.x = "dest", by.y = "iso3", all.x = TRUE)
setnames(master_dt, "country_code", "cod_d")

# Jointure Origine
master_dt <- merge(master_dt, country_stats[, !"iso3"],
                   by.x = c("cod_o", "year0"), by.y = c("country_code", "year"),
                   all.x = TRUE)
setnames(master_dt,
         c("P_t","psr","IMR_t","LA","LL","urban_t","PIB","PIB_lag1","PIB_lag5"),
         c("P_it","PSR_i","IMR_it","LA_i","LL_i","urban_it",
           "gdp_o","gdp_o_lag1","gdp_o_lag5"))

# Jointure Destination
master_dt <- merge(master_dt, country_stats[, !"iso3"],
                   by.x = c("cod_d", "year0"), by.y = c("country_code", "year"),
                   all.x = TRUE)
setnames(master_dt,
         c("P_t","psr","IMR_t","LA","LL","urban_t","PIB","PIB_lag1","PIB_lag5"),
         c("P_jt","PSR_j","IMR_jt","LA_j","LL_j","urban_jt",
           "gdp_d","gdp_d_lag1","gdp_d_lag5"))

# Jointure CEPII avec la table enrichie
master_dt <- merge(master_dt, dist_clean,
                   by.x = c("orig", "dest"), by.y = c("iso_o", "iso_d"),
                   all.x = TRUE)

# variables dérivées


master_dt[, `:=`(
  year         = year0,
  t_2000       = year0 - 2000,
  t_2000_sq    = (year0 - 2000)^2,
  is_migration = as.integer(flow > 0)
)]

# GDP par tête :les deux lags
master_dt[, `:=`(
  gdpcap_o      = gdp_o      / P_it,
  gdpcap_d      = gdp_d      / P_jt,
  gdpcap_o_lag1 = gdp_o_lag1 / P_it,   # lag t-1 (haute fréquence)
  gdpcap_d_lag1 = gdp_d_lag1 / P_jt,
  gdpcap_o_lag5 = gdp_o_lag5 / P_it,   # lag t-5 (quinquennal )
  gdpcap_d_lag5 = gdp_d_lag5 / P_jt
)]

# Renommage pour compatibilité avec le notebook Python existant
# (log_gdpcap_o_lag pointe sur lag5 — spécification originale conservée)
master_dt[, `:=`(
  gdpcap_o_lag  = gdpcap_o_lag5,
  gdpcap_d_lag  = gdpcap_d_lag5
)]

# Logs
log_vars <- c("gdp_o","gdpcap_o","gdp_d","gdpcap_d",
              "gdp_o_lag1","gdpcap_o_lag1","gdp_d_lag1","gdpcap_d_lag1",
              "gdp_o_lag5","gdpcap_o_lag5","gdp_d_lag5","gdpcap_d_lag5",
              "gdpcap_o_lag","gdpcap_d_lag")
master_dt[, (paste0("log_", log_vars)) := lapply(.SD, log), .SDcols = log_vars]


#  EXPORT


final_cols <- c(
  "orig","dest","year","flow","P_it","PSR_i","IMR_it","urban_it","LA_i","LL_i",
  "P_jt","PSR_j","IMR_jt","urban_jt","LA_j","LL_j",
  "D_ij","LB_ij","OL_ij","COL_ij","t_2000","t_2000_sq","is_migration",
  "gdp_o","gdpcap_o","gdp_d","gdpcap_d",
  "gdp_o_lag1","gdpcap_o_lag1","gdp_d_lag1","gdpcap_d_lag1",
  "gdp_o_lag5","gdpcap_o_lag5","gdp_d_lag5","gdpcap_d_lag5",
  "gdpcap_o_lag","gdpcap_d_lag",   # alias lag5 pour compatibilité Python
  "log_gdpcap_o","log_gdpcap_d",
  "log_gdpcap_o_lag1","log_gdpcap_d_lag1",
  "log_gdpcap_o_lag5","log_gdpcap_d_lag5",
  "log_gdpcap_o_lag","log_gdpcap_d_lag"
)
final_cols_present <- intersect(final_cols, names(master_dt))

df_final <- master_dt[, ..final_cols_present]

fwrite(df_final,
       file = "/Users/romain/Desktop/Projets DS/ProjetStat/data/panel_june_R.csv",
       sep = ",", dec = ".", row.names = FALSE, col.names = TRUE)

cat("Export terminé :", nrow(df_final), "lignes,", ncol(df_final), "colonnes\n")

# Audit rapide post-export
cat("\nPays avec D_ij manquant dans le final :\n")
print(unique(df_final[is.na(D_ij), .(orig, dest)])[1:20])
