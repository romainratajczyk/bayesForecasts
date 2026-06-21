
# Prédiction bayésienne des flux migratoires internationaux 

Projet de recherche supervisé par Nicolas Chopin (CREST), réalisé à l'ENSAE.   


# TL;DR :  

This research project develops a Bayesian architecture for the short-term prediction (up to 5 years) of global bilateral migration flows. We start from the use of a simple log-linear gravity model, and the use of ensemble algorithms (Random Forest, XGBoost) allowing us to map the spatial heterogeneity of errors and to isolate geographic or macroeconomic non-linearities (concave effect of distance or GDP threshold effects and poverty traps discussed by A.Banerjee and E.Duflo in Good Economics for Hard Times). To mitigate the vulnerability to short-term macro-geopolitical shocks of the reference Bayesian model (Welch & Raftery, 2022)—whose structure remains deliberately parsimonious and elegantly robust for long-term projections—then we introduce a new hierarchical architecture: the Hurdle ARX-ZTNB model, estimated via HMC-NUTS (with Stan software). The 51% proportion of zeros in the global flow matrix invalidates the use of a unimodal distribution. Our architecture therefore separates the existence of a corridor from its intensity, two orthogonal processes (crucial) that make the Fisher information matrix block-diagonal. On the one hand, a logit regression using covariates derived from non-linear machine learning discriminates the opening of corridors with high precision. On the other hand, the intensity of strictly positive flows is governed by an ARX(1) component capturing migratory hysteresis, coupled with a Zero-Truncated Negative Binomial (ZTNB) distribution. This discrete distribution natively absorbs the strong dispersion of flows around the mean and handles variance heterogeneity hierarchically, and by M49 world region (a sub-continental set). Out-of-sample evaluation confirms that the performance gain justifies the complexity: a decrease in absolute error (MAE at 1159 migrants), a more confident and robust model (95% CI coverage at 95.9%), and a massive 30-point reduction in relative error (MAPE 46.9%) which may be of interest to policymakers. 
  
# Annexe technique : Bayesian Hierarchical ARX Hurdle Model (notre modèle de prédiction court-terme) (à jour de Mars 2026 - le modèle a changé depuis)

Cette section détaille l'architecture mathématique et les choix d'inférence de notre modèle bayésien. Pour ceux qui souhaitent comprendre le moteur interne de notre code Stan et la méthodologie de prédiction.

### 1. Architecture en deux étapes (Hurdle-Volume)

Le modèle traite la migration bilatérale en deux étapes séquentielles pour contourner la double difficulté des flux nuls (49% du dataframe) et de la forte variance des grands couloirs.

#### A. Composante Hurdle (Proba d'Ouverture de la route)
Régression logistique (Bernoulli) estimant la probabilité qu'un flux migratoire strictement positif existe entre les pays $i$ et $j$.

$$\text{logit}(P(\text{flow} > 0)) = \alpha_{d} + X_{h} \beta_{h} + \beta_{\text{lag}} \text{is\\_mig\\_lag}$$

Où $X_{h}$ inclut les variables les plus importantes et pertinentes pour le Hurdle (notamment les features les plus importantes indiquées par un Random Forest entraîné) : frontière commune, $\log(\text{distance})$, PIB/tête à la date $t-1$, populations... Sans pour autant répliquer complètement le modèle de gravité (le but est l'*existence ou non* d'une route, pas son *volume*). Si le modèle prédit une fermeture, le flux prédit est 0 net. S'il prédit une ouverture, on passe à la composante Volume.

#### B. Composante Volume (Processus ARX Log-Normal)
### Précision: la distribution log-normale a été remplacée par une Negative Binomiale tronquée en zéro (ZTNB). 
AR "X" pour "eXogenous variables", les variables économétriques du modèle de gravité pour $$\mu$$.   
Pour les dyades actives, le volume est modélisé par un processus auto-régressif conditionnel à la dyade :

$$\log(\text{flow}) \sim \mathcal{N}(\mu_{d,t} + \phi_{d} (\text{lag} - \mu_{d,t-1}), \sigma_{d})$$

L'espérance de base $\mu_{d,t}$ intègre les variables gravitaires classiques et les variables non-linéaires découvertes lors de la phase d'exploration par Machine Learning :

$$\mu_{d,t} = \alpha_{V,d} + X \beta_{\text{grav}} + \beta_{\text{gdp}} \log(\text{gdpcap\\_o}) + \beta_{\text{rich}} \text{is\\_rich\\_o}$$

*(Note : `is_rich_o` encode un effet de seuil détecté par Random Forest autour de 18 000 $ de PIB/habitant à partir duquel l'émigration augmente brusquement pour le pays d'origine).*

### 2. Inférence par Hamiltonian Monte Carlo (HMC) avec Stan

Contrairement aux approches par échantillonnage de Gibbs (JAGS) ou marche aléatoire aveugle (Metropolis), l'utilisation de Stan (HMC) est cruciale ici pour explorer un espace de paramètres de très haute dimension (~90 000 dimensions) sans rester piégé.

**Le paysage énergétique et la mécanique hamiltonienne**
L'espace des postérieurs bayésiens est analogue à un paysage énergétique en physique où la log-vraisemblance définit l'énergie potentielle (les "puits" sont les zones de forte probabilité ici). À chaque itération $s$ :
1. L'algorithme reçoit une impulsion cinétique aléatoire.

2. Il simule une trajectoire déterministe le long du gradient de probabilité via les équations de Hamilton. Au moment d'une micro-étape (itération $s$), le moteur Stan fait concrètement ceci :
   * Il prend les valeurs `raw` tirées du bruit (des priors qui peuvent être faiblements informatifs, ou légèrement calibrés par Empirical Bayes) et les multiplie par les $\tau$ pour construire l'état de chaque couloir : $\alpha_{V,d}^{(s)}$, $\phi_{d}^{(s)}$ et $\sigma_{d}^{(s)}$.
   * Il assemble tout ça avec les variables géoéconomiques ($X$, PIB, etc.) pour calculer le $\mu_{d,t}^{(s)}$.
   * Puis, il utilise cette valeur pour évaluer la distance par rapport aux vrais flux via la loi Volume :

   $$\log(\text{flow}) \sim \mathcal{N}(\mu_{d,t}^{(s)} + \phi_{d}^{(s)} (\text{lag} - \mu_{d,t-1}^{(s)}), \sigma_{d}^{(s)})$$

3. À la position d'arrivée, Stan évalue l'acceptation via Metropolis-Hastings en vérifiant la conservation de l'énergie totale ($H$) :

   $$P(\text{acceptation}) = \min(1, \exp(-\Delta H))$$

Si la position est cohérente ($\Delta H \approx 0$), les paramètres sont acceptés et inscrits dans les chaînes de Markov.

**Stabilité géométrique (Non-centered parameterization)**
Pour éviter les géométries en entonnoir qui font diverger/bloquent les chaînes de Markov, le modèle hiérarchique est codé via une paramétrisation décentrée (*transformed parameters*). Stan ne tire pas directement dans la loi normale de la dyade, il tire un bruit pur (`raw`) qu'il multiplie par la variance du cluster ($\tau$) :
* **Intercept dyadique :**

  $$\alpha_{V,d} = \mu_{\text{intercept}} + \tau_{\mu} \times \mu_{\text{raw}}[d]$$

* **Inertie AR1 :**

  $$\phi_{d} = \tanh(\phi_{\text{global} \_ \text{raw}} + \tau_{\phi} \times \phi_{\text{raw}}[d])$$

* **Variance hétéroscédastique :**

  $$\sigma_{d} = \sigma_{\text{cluster}} \times \exp(\tau_{\sigma} \times \sigma_{\text{raw}}[d])$$
  
### 3. Méthode de prédiction

Une fois l'inférence terminée, les matrices de paramètres (ex: 1200 itérations conservées, entraînement sur 1990-2010) sont extraites. NumPy prend la relève pour vectoriser les équations sur les données hors-échantillon (ex: test sur 2015).

**Le choix de la Médiane vs l'Espérance**
Dans un modèle log-normal, l'espérance mathématique est $\exp(\mu + \sigma^2 / 2)$. Sur des couloirs instables (comme MEX-USA), il a été observé un grand $\sigma_{d}$ amplifié par l'inflation auto-régressive $(1+\phi^2)$ ce qui a propulsé les prédictions à des valeurs absurdes (ex: 25 millions de migrants) en tentant de minimiser la *Mean Squared Error* (MSE).

Or, l'objectif macroéconomique et décisionnel est de minimiser l'erreur absolue en nombre d'humains, pas en humains au carré. Ainsi nous extrayons la médiane $\exp(\mu)$ de nos matrices de prédiction, qui est le minimiseur naturel de la norme L1 (MAE).

### 4. Choix méthodologiques et Discussion

* **Synergie ML $\rightarrow$ Bayésien :** Le modèle bayésien n'est pas construit à l'aveugle. Il intègre directement les enseignements de nos modèles XGBoost et Random Forest : effets de seuils sur le PIB, interactions spatiales validées par PDP ($\log(\text{Distance}) \times \text{Frontière}$), et hétéroscédasticité géographique modélisée au niveau continental pour absorber les résidus systématiques détectés en Afrique et en Asie (sur des cartes de résidus mondiales, cf `challenge_gravity_ML.ipynb`).
* **Le problème des zéros :** L'approche Hurdle a été préférée à la transformation $\log(x+1)$ (qui est scientifiquement instable). Forcer une loi normale continue à gérer un pic massif à zéro provoque une divergence de la variance temporelle. Le Hurdle isole le problème structurellement.
* **Évaluation (OOS) :** Entraîné sur la période 1990-2010 et testé sur 2015. Nous utilisons le WMAPE, et la MAPE modifiée de Welch & Raftery (divisée par $y+1$) pour un benchmark fidèle face à la littérature (Welch & Raftery). La couverture spatiale des intervalles de crédibilité (IC) bénéficie beaucoup de l'hétéroscédasticité : étroits en Europe (+/- 30%), ils s'élargissent logiquement sur les couloirs volatiles d'Asie et d'Afrique (+/-150%).
* **La limite des micro-flux :** Le modèle présente un biais théorique inhérent à la loi log-normale sur les flux continus de 1 à 10 migrants. Si un modèle de comptage (ex: Negative Binomial) traiterait mieux ces micro-flux, l'ajout d'un modèle pour les flux intermédiaires nous semble trop *ad-hoc* et perturberait certainement la stabilité de nos simulations. Surtout, ces micro-flux sont macro-économiquement non pertinents et résultent de bruit statistique : on assume alors que notre modèle n'est pas adapté à la prédiction sur les micro-flux.

### 5. Dimensions de l'espace des paramètres  

L'inférence simultanée repose sur une très-haute-dimension (pour 190 pays) :
* **Partie Hurdle ($D_{h}$) :** $\sim 35\ 000$ dimensions ($\alpha_{\text{raw}}$ par dyade).
* **Partie Volume ($D_{v}$) :** Environ 50% des dyades sont actives. Chacune requiert un $\mu_{\text{raw}}$, un $\phi_{\text{raw}}$ et un $\sigma_{\text{raw}}$, soit $\sim 53\ 000$ dimensions.
* **Paramètres globaux & Clusters :** Vecteurs $\beta_{h}$ (3 variables), $\beta_{\text{grav}}$ (~20 variables), variances par continent (6 dimensions), et hyper-paramètres globaux ($\mu$, $\tau$).

**Total : $\sim 90\ 000$ dimensions explorées simultanément par Hamiltonian Monte Carlo.**
*Estimation RAM : 50-64 Go pour être très confortable et robuste aux pics et aux "Silent Kills" du cluster Onyxia-GENES. Plus de 128 Go nécessaires pour extraire TOUTES les variables samplées par Stan (pour 190 pays), le code actuel ne retire que celles importantes pour les prédictions.*    



*Dernière mise à jour : 10 Avril 2026*

