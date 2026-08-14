# =============================================================================
# fig:posterior_pij — distributions postérieures de la probabilité d'ouverture
# Entrée : posterior_pij_<RUN>.parquet produit par cell_posterior_pij.py
# =============================================================================
install.packages(c("arrow", "dplyr", "ggplot2", "ggdist"))
library(arrow); library(dplyr); library(ggplot2); library(ggdist)

RUN   <- "t300_k2.0_pw1.0"
SEUIL <- 0.5

d <- read_parquet(sprintf("/Users/rratajczyk/Desktop/bayesForecasts/notebooks/posterior_pij_t300_k2.0_pw1.0.parquet", RUN)) |>
  mutate(
    statut   = factor(ouvert_2015, levels = c(0, 1),
                      labels = c("corridor fermé en 2015", "corridor ouvert en 2015")),
    corridor = reorder(corridor, p, FUN = median)
  )

p <- ggplot(d, aes(x = p, y = corridor, fill = statut)) +
  stat_halfeye(.width = c(0.50, 0.95), point_interval = "median_qi",
               slab_alpha = 0.80, adjust = 1.2, normalize = "groups") +
  geom_vline(xintercept = SEUIL, linetype = "dashed", linewidth = 0.3) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0),
                     breaks = seq(0, 1, 0.25),
                     name = expression(p[ij]~"(probabilité postérieure d'ouverture)")) +
  scale_fill_grey(start = 0.35, end = 0.75, name = NULL) +
  labs(y = NULL) +
  theme_minimal(base_size = 10) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        legend.position = "bottom")

ggsave("Graphiques/posterior_pij.pdf", p,
       device = cairo_pdf, width = 6.5, height = 4.0)