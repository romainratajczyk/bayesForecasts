# M3 : N agents (oligopolistes / cartellistes / frange), T periodes, capacite active, PINN.
# Objectif : equilibres de Markov parfaits (MPE) closed-loop a N > 1 joueurs STRATEGIQUES,
# avec graphe d'influence (DAG) : oligopolistes singletons, cartels, frange price-taker,
# capacite kappa active, dynamique hotellingienne.
#
# ACQUIS, ET POURQUOI ILS TIENNENT
#
# 1. A UN SEUL JOUEUR STRATEGIQUE, LA DIVERGENCE CL/OL N'EXISTE PAS. Benchekroun &
#    Withagen (2012, Games and Economic Behavior 76(2), 355-374) : les issues du jeu
#    cartel-frange en boucle ouverte et en boucle fermee COINCIDENT quand la frange est
#    preneuse de prix (jeu a la Salant 1976). La litterature sur l'incoherence temporelle
#    porte sur von STACKELBERG, ou le cartel internalise la reaction INSTANTANEE de la
#    frange (dx_f/dx_c). Ce fichier implemente h_i = p - beta*Lam_i - c_i - delta*lambda'_i,
#    donc du Nash-Cournot : feedback = 0 est le resultat ATTENDU, pas un echec.
#    -> il faut >= 2 STRATEGIQUES. C'est la calibration par defaut (a0, a1 + frange).
#
# 2. LE FEEDBACK NE SE LIT PAS DATE PAR DATE. La chaine d'enveloppe donne
#        lambda_i(0) = somme_t delta^t fb_i(t) + delta^T lambda_i(T),
#    et comme lambda croit exactement au taux r, delta^t * fb_i(t) est CONSTANT des lors
#    que fb_i/lambda_i l'est : chaque date interieure contribue le MEME montant, sans
#    attenuation. 0.3% par date sur 20 dates -> ~6% sur mu.
#    MESURE sur l'oracle seul (aucun reseau) : a0 +1.69%, a1 +5.58%. C'est la PREDICTION
#    QUANTITATIVE que l'etage C doit reproduire. Voir CFG["fb_ref"].
#
# 3. LE FEEDBACK EST NUL DES QU'UN STRATEGIQUE EST A UN COIN (J[k,j] = 0). Il faut donc
#    des dates ou DEUX strategiques sont SIMULTANEMENT INTERIEURS. L'ecran ETAPE 1b le
#    verifie avant tout entrainement : 20 dates ici.
#
# 3bis. LA CIBLE DU CO-ETAT EST CLAMPEE A ZERO **APRES** L'AJOUT DU FEEDBACK.
#    lambda = softplus(.)*LREF*G est strictement positif : une cible negative est
#    inatteignable, R_env reste positif en permanence et pousse softplus -> 0. Le feedback
#    etant negatif sur ~37% des points non nuls, clamper la seule marge strategique avant
#    de lui ajouter le feedback laissait passer ce cas.
#
# 4. LA CIBLE DU CO-ETAT DOIT ETRE DURE EN REGIME INTERIEUR (env_hard). Un max(dur,
#    delta*lambda') est un CLIQUET VERS LE HAUT : si le reseau surestime lambda', le max
#    selectionne le bootstrap et court-circuite la cible dure exactement quand elle
#    servirait. Correctif : utiliser le regime, deja connu exactement par `bind`.
#
# 5. LE VALIDATEUR QUI SURVIT A LA DISPARITION DE L'ORACLE, c'est STEP 9, et en son coeur
#    le TEST DE DEVIATION A UN COUP : un profil est un equilibre ssi aucun joueur ne gagne
#    a devier une fois puis a reprendre sa politique. Rivaux GELES -> condition OPEN-LOOP ;
#    rivaux qui REPONDENT -> condition MARKOVIENNE. L'ecart entre les deux gains EST la
#    divergence CL/OL, sans oracle ni grille.
#
# LE MODE D'ECHEC A NE PLUS REPETER : mesurer avec un instrument qu'on n'a pas d'abord
# valide contre une verite. REGLE : tout nouvel objet (gV hors-diagonale, J, feedback,
# grille) passe son propre test de verite AVANT de servir de mesure.
#
# ETAGES (toggles cross / sob_off / feedback_only)
#   A. cross=False, sob_off=False : systeme OPEN-LOOP. L'oracle est la VERITE.
#      Attendu : deviation FREEZE ~ 0.
#   B. cross=False, sob_off=True  : on entraine aussi le gradient hors-diagonale via
#      R_soff. OPTIONNEL : 9f a montre que gV croise est deja bon a 7.5% SANS R_soff,
#      dont la cible est auto-referentielle. Ne l'activer que si 9f se degrade.
#   C. cross=True, sob_off=True, feedback_only=True : MPE. On CALCULE J, GV2 et feedback
#      sans les ENTRAINER (R_sob et R_soff annules). L'etage C devient l'etage A plus une
#      ligne : l'injection du feedback dans la cible du co-etat.
#      Attendu : deviation MARKOV ~ 0, deviation FREEZE en HAUSSE, rente deplacee de
#      fb_ref, et 9b Euler s'ecartant de (1+r) de fb/lambda.
#
# COUT : sob_off/cross demandent 3*NAG backward passes avec create_graph.


# STEP 0 : imports, config, constantes derivees
import os, sys, time, json
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)

CFG = dict(
    tag="m3_n3_base_ckpt", seed=0,
    sampler="mix",           # "mix" = tube+uniforme (defaut) | "tube_oracle" = bras degrade

    # ECONOMIE : listes paralleles, un element par AGENT
    # DEFAUT N=3 : deux oligopolistes singletons (blocs 0 et 1) + une frange (-1).
    #   20 dates ou a0 ET a1 sont simultanement interieurs -> J != 0 -> feedback non nul.
    # Non-regression M2 (N=2) : costs="10,20" kappas="20,30" stocks="700,450" blocs="0,-1"
    #   -> CLOS : coincidence CL/OL demontree et verifiee. Utile comme test de socle.
    costs  = "10,12,20",
    kappas = "18,14,25",
    stocks = "620,410,400",
    blocs  = "0,1,-1",       # entiers distincts = oligopolistes ; egaux = meme cartel ; -1 = frange
    alpha=100.0, beta=1.0, r=0.05, T=50,

    # ETAGES
    sob_off=True,            # calculer J, GV2, feedback (necessaire des que cross=True)
    cross=False,             # injecter le feedback dans la cible du co-etat => MPE
    # LIGNE DE BASE. cross=False donne le POINT ZERO de STEP 10 : le reseau a un biais de
    # niveau contre mu MEME sans feedback (+0.8% sur a0), donc (niveau - mu)/mu additionne
    # le deplacement cherche et ce biais. Ce run doit differer du run etage C par cross
    # SEULEMENT : soff_mode, env_hard, use_vmc, iters et la graine restent identiques.
    # feedback_only : calculer J/GV2/feedback SANS entrainer R_sob ni R_soff.
    #   9f a mesure gV croise a 7.5% sans R_soff, donc R_soff n'apporte rien ici et porte
    #   tout le risque de sa cible auto-referentielle. L'etage C se reduit alors a
    #   l'etage A + l'injection du feedback : un seul changement, risque minimal.
    #   Mettre False pour reactiver le Sobolev complet (etage B classique).
    # soff_mode : COMMENT superviser le gradient hors-diagonale a l'etage C.
    #   "pathwise" (DEFAUT) : cible MESUREE par differences centrees sur des rollouts de la
    #      politique courante (gv_target_fd). Non auto-referentielle. C'est la bonne
    #      reponse a l'objection "sans supervision explicite, gV derive quand la surface
    #      bascule de OL vers CL". Cout : +25-30% de temps.
    #   "off" : ne superviser que via R_lam et le niveau de V. Mesure a l'etage A :
    #      gV croise 7.5%, diagonal 1.2%/0.7%. Suffisant SI 9f le confirme a l'etage C.
    #   "bellman" : ancienne cible delta*GV2 + M_full, construite depuis le V du reseau.
    #      A converge vers un point fixe auto-coherent mais FAUX (28.9% sur la diagonale
    #      de la frange a N=2). Conserve pour bissection, deconseille.
    # RUN A a TESTE "off" et la reponse est NON : la supervision pathwise AIDE.
    #   avec  : err_p_noext 9.3e-3 | field_int_med 0.92 | J_diag_med 16.5 % | gV croise 11.9 %
    #   sans  : err_p_noext 1.4e-2 | field_int_med 1.52 | J_diag_med 30.4 % | gV croise 15.3 %
    # La comparaison avec l'etage A2 (7.5 % sans pathwise) ne transferait pas : version de
    # code anterieure. Ne pas repasser a "off" sans argument neuf.
    soff_mode="pathwise",
    gv_pts=256,              # points du buffer sur lesquels la cible pathwise est calculee
    gv_batch=128,            # sous-echantillon utilise a chaque iteration
    gv_every=400,            # frequence de rafraichissement de la cible

    # TOGGLES DE BISSECTION (RUN 9). Ne jamais en bouger deux dans le meme run.
    sob_ok_mask=True,        # SOB_OK = lignes canoniques de A (False -> ones : RUN 6, casse)
    G_in_V=True,             # facteur G(t) dans la construction de V
    norm_G_bell=True,        # R_bell divise par G_t
    norm_G_lam=True,         # R_env/R_sob/R_lam divises par G_t
    vref_per_agent=False,    # VREF vectoriel alpha*S0_i au lieu du scalaire alpha*max(S0)

    # Cible Monte-Carlo TD(1) pour V, sur le bras TUBE PUR uniquement. VALIDE.
    #   R_bell est un residu a UN pas : il peut etre minuscule pendant que V derive de 25%.
    #   Le buffer fait deja des rollouts complets, la cible est donc gratuite et exacte.
    use_vmc=True, w_vmc=1.0,

    # Cible DURE pour lambda en regime INTERIEUR (voir acquis 4 en en-tete). VALIDE :
    # a supprime le biais de niveau (+2.1% -> +0.8%) et divise par 4 l'erreur aux bascules.
    env_hard=True,

    # Amplification du feedback injecte dans la cible du co-etat (action 2).
    # 1.0 = STRICTEMENT INERTE. Passer a 10.0 pour discriminer trois issues distinctes :
    #   deplacement de rente x10 -> canal lineaire et fonctionnel, corriger J ;
    #   aucun deplacement       -> canal casse, chercher dans l'injection/le detachement ;
    #   deplacement erratique   -> feedback domine par du bruit de signe, confirme J.
    fb_gain=1.0,

    # R_dfoc : SOBOLEV SUR LA FOC (action 3). INERTE tant que dfoc=False.
    #   A l'equilibre interieur h_i(S) == 0 sur un OUVERT, donc dh_i/dS_j = 0. C'est le
    #   SEUL residu qui contraigne J HORS-DIAGONALE, mesure a 99.5% d'erreur mediane en 9h
    #   et nul aux dates precoces. Cout : +N passes arriere (4N au lieu de 3N).
    #   FAIBLESSE A SURVEILLER : une equation, DEUX objets libres (J et L = dlam'/dS').
    #   L'optimiseur peut l'annuler en deformant L au lieu de corriger J. Le juge n'est
    #   donc PAS le residu mais 9h : J_offdiag_med doit passer de 99.5% a moins de 20%.
    #   Sequence : valider a cross=False d'abord, puis seulement a cross=True.
    #   RESULTAT DU RUN 2 : ECHEC CATASTROPHIQUE, et pas a cause du poids.
    #   L'equation dh_i/dS_j = 0 admet une solution TRIVIALE -- une politique qui ne
    #   depend pas de l'etat -- que l'architecture rend GRATUITE (x = kappa est atteint a
    #   la precision machine). La descente de gradient l'a trouvee : politique bang-bang
    #   collee a kappa, J = 0.00000 partout (diagonale comprise), rente ecrasee a -7 %,
    #   err_p_noext de 9.3e-3 a 7.4e-2. c_dfoc a represente >90 % de la loss sans jamais
    #   descendre (8 a 10 de it=3000 a it=18000) : un cout paye, pas un residu minimise.
    #   Le bassin degenere existe a TOUT poids > 0 ; le poids ne regle que la vitesse de
    #   chute. Code conserve car inerte a dfoc=False, et parce qu'il documente l'impasse.
    dfoc=False, w_dfoc=1.0,

    # Ecran de feedback sur l'ORACLE seul. DESACTIVE : son travail est fait, le resultat
    # est fige dans fb_ref ci-dessous. Remettre True si la calibration change.
    run_fb_screen=False, fb_h=1e-2,
    # TEST DE VERITE de fb_ref (action 0). fb_ref sert de reference a STEP 10 sans que sa
    # sensibilite au pas de difference finie ait jamais ete testee. Tant que ce balayage
    # n'est pas passe, le chiffre le plus solide du projet est un instrument non valide,
    # exactement comme 9h avant sa correction.
    # Balayage effectue : a1 stable a 1.4% (5.52/5.58/5.60), a0 a 15% et monotone
    # decroissant. Test de verite passe pour a1, marginal pour a0. Inutile de le refaire.
    fb_h_sweep=False, fb_h_list="5e-3,1e-2,2e-2",
    # PREDICTION MESUREE de l'ecart CL/OL sur la rente, un element par agent, en fraction
    # de mu. C'est le juge quantitatif de l'etage C (STEP 10).
    fb_ref="0.0169,0.0558,0.0",

    # ETAPE 1b : ecran de calibration (cout : quelques secondes, aucun entrainement)
    run_screen=True,

    # ETAPE 2 : grille MPE (N = 2 ou 3). Cout ~O(ng^N * nx * nbr * T).
    #   N=2 : ng=160 nx=31 -> ~30 s.   N=3 : ng=32 nx=15 -> ~2-4 min.
    #   La grille n'est PAS le validateur principal (sa precision plafonne vers 0.5%
    #   en 2D et bien au-dela en 3D). C'est STEP 9 qui valide. La grille sert de
    #   controle croise independant.
    run_mpe_grid=False, mpe_ng=32, mpe_nx=15, mpe_nbr=6,

    # reseau
    width=128, emb_dim=16, depth=3,

    # loss / sampler
    w_sob=0.01, w_sob_off=0.1, frac_unif=0.05, frac_gauss=0.25, frac_decor=0.0, box=1.25,
    buf_paths=64, buf_every=100, buf_lo=0.70, buf_hi=1.25,

    # entrainement
    # ACCELERATION : la loss plafonne vers 15-18k iterations, 20000 suffit. L-BFGS ne
    # converge pas des que sob_off=True (loss_pre == loss_post, Wolfe echoue au premier
    # pas) : le mettre a 0. buf_every 100 au lieu de 50 : ~5% de gain, effet negligeable
    # sur le tube. Total ~-45% de temps par rapport a la configuration etage A.
    iters=20000, batch=512, lr=1e-3, lbfgs_outer=0,

    # A 0 pour ce run : grad_norms tire un batch, donc il consomme du RNG et decale le
    # flux d'echantillonnage. Le juge de ce run est une COMPARAISON, elle doit rester nette.
    diag_every=0,            # instrumentation R9bis : decomposition loss + |grad| par residu

    # CHECKPOINT. Un state_dict est le dictionnaire des poids et biais du modele (pas
    # l'architecture, pas l'optimiseur). Le sauver permet d'iterer sur les DIAGNOSTICS en
    # secondes au lieu de reentrainer 40 min a chaque question. Le rechargement suppose
    # width / depth / emb_dim / NAG identiques, sinon load_state_dict leve une erreur.
    load_ckpt=False,         # True : recharger ckpt_<tag>.pt et SAUTER l'entrainement

    # 9h-bis : J ANALYTIQUE, resolu au lieu d'etre lu dans la tete politique.
    # Diagnostic pur, aucun effet sur l'entrainement.
    s9_Jana=True,

    # eval
    snap_eps=1e-4, n_field=24, or_nbis=30, or_nouter=15,
    t_decouple="10,20,30", make_figs=True,

    # STEP 9 : batterie de coherence SANS ORACLE
    run_step9=True,
    dev_eps_rel=0.15,        # amplitude de deviation, en fraction de la capacite du pas
    dev_n_eps=13,            # points de la grille de deviation (impair : 0 au centre)
    dev_n_dates=5,           # dates sondees
    dev_skip_ext=True,       # exclure les dates d'extinction : sinon le max est pilote
                             # par elles, exactement le defaut deja corrige sur err_p
    fd_rel=5e-3,             # pas relatif des differences finies (gV, J)
    s9_J=False,              # 9g : cablage de J (autodiff vs FD du reseau). Valide 4.7e-5.
    s9_Jor=True,             # 9h : J du reseau vs J de l'ORACLE. Teste l'ECONOMIE, pas le
                             # cablage. C'est le test qui manquait : 9g ne compare le
                             # reseau qu'a lui-meme.
    s9_Jor_dates=6,          # 9h porte desormais sur le HORS-DIAGONAL : plus de dates,
                             # cout marginal (2N resolutions d'oracle par date)
)

for a in sys.argv[1:]:
    if "=" not in a: continue
    k, v = a.split("=", 1)
    if k not in CFG: continue
    CFG[k] = (v.lower() in ("1", "true")) if isinstance(CFG[k], bool) else type(CFG[k])(v)

alpha, beta, r, T = CFG["alpha"], CFG["beta"], CFG["r"], CFG["T"]
delta = 1.0/(1.0+r)
C_np   = np.array([float(u) for u in CFG["costs"].split(",")])
KAP_np = np.array([float(u) for u in CFG["kappas"].split(",")])
S0_np  = np.array([float(u) for u in CFG["stocks"].split(",")])
BLOC   = np.array([int(u)   for u in CFG["blocs"].split(",")])
NAG = len(C_np)
assert len(KAP_np) == len(S0_np) == len(BLOC) == NAG, "listes d'agents de longueurs differentes"

# masque de conduite : A[i,j] = 1 ssi i et j sont dans le meme bloc (>=0). Frange : ligne nulle.
# cartelliste -> Lambda_i = Q^C (somme du bloc) ; oligopoliste -> Lambda_i = x_i ; frange -> 0.
A_np = np.zeros((NAG, NAG))
for i in range(NAG):
    if BLOC[i] < 0: continue
    for j in range(NAG):
        if BLOC[j] == BLOC[i]: A_np[i, j] = 1.0
IS_FRINGE = BLOC < 0

# SOB_OK : masque de R_lam SEULEMENT (l'identite dV_i/dS_i = lambda_i). Cette identite
# n'est valide que si coef[i,i] = 0, c'est-a-dire pour un OLIGOPOLISTE SINGLETON a
# l'optimum interieur (theoreme de l'enveloppe : le terme de choix propre s'annule).
#   singleton interieur : coef_ii = p - c_i - beta*x_i - delta*GV2_ii = 0
#   cartelliste         : coef_ii = beta*(Q^C - x_i) != 0
#   frange              : coef_ii = -beta*x_f        != 0
# R_sob, l'enveloppe universelle derivee de Bellman, n'a jamais eu de masque et n'en veut
# pas. RUN 6 (SOB_OK = ones) imposait deux equations incompatibles au meme objet. Ne pas
# y revenir.
SOB_OK = (torch.tensor((A_np == np.eye(NAG)).all(1).astype(float))
          if CFG["sob_ok_mask"] else torch.ones(NAG))

if CFG["cross"] and not CFG["sob_off"]:
    print("[!] cross=True force sob_off=True (le feedback depend de dV_i/dS_k)")
    CFG["sob_off"] = True

torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
A    = torch.tensor(A_np)
COST = torch.tensor(C_np)
KAP  = torch.tensor(KAP_np)
S0V  = torch.tensor(S0_np)
DEN  = beta*(1.0 + torch.diagonal(A))   # pas de Newton du clamp : dh_i/dx_i = -beta(1+A_ii).
                                        # Ce n'est PAS un parametre de conduite : la conduite
                                        # est entierement dans Lambda = A x.
OFF  = 1.0 - torch.eye(NAG)             # masque k != i
VREF = (alpha*S0V) if CFG["vref_per_agent"] else (alpha*S0_np.max())
LREF = alpha                            # rente bornee par le choke price
XREF = torch.minimum(KAP, S0V)          # echelle des QUANTITES (pas des stocks)
W_SOB, W_SOFF = CFG["w_sob"], CFG["w_sob_off"]
W_DFOC = CFG["w_dfoc"]
FRAC_UNIF, FRAC_GAUSS, BOX, SNAP_EPS = CFG["frac_unif"], CFG["frac_gauss"], CFG["box"], CFG["snap_eps"]

# STRAT[i] = 1 pour un agent STRATEGIQUE (ligne non nulle dans A), 0 pour la frange.
# Une frange preneuse de prix n'a pas de terme de feedback. VALIDE au RUN 13b.
STRAT = torch.tensor((~IS_FRINGE).astype(float))
STRAT_IDX = [i for i in range(NAG) if not IS_FRINGE[i]]

print(f"STEP 0 : {NAG} agents | blocs={list(BLOC)} | strategiques={STRAT_IDX} "
      f"| frange={list(np.nonzero(IS_FRINGE)[0])}")
print(f"         A =\n{A_np.astype(int)}")

def growth_t(ti):
    """G(t) = (1+r)^(t-(T-1)) : vaut 1 a la derniere date de decision, ~0.09 en t=0.
       C'est EXACTEMENT le facteur de croissance de Hotelling : lambda_i(t) = mu_i(1+r)^t.
       Ecrire lam = softplus(l)*LREF*G ramene donc la cible de la tete co-etat a une
       CONSTANTE en t le long du sentier. Preconditionnement exact, adosse a la theorie."""
    return (1.0+r)**(ti.double() - (T-1))


# STEP 1 : oracle open-loop a N joueurs [INCHANGE, valide analytiquement]
# rentes initiales mu_i, lambda_i(t) = mu_i*(1+r)^t. A rentes donnees, chaque periode est un
# equilibre STATIQUE a couts effectifs c_i + lambda_i(t) (gere les coins nativement).
# On cherche N scalaires au lieu de N*T inconnues et 3^(N*T) regimes.
# Trois boucles : (a) point fixe de meilleures reponses intra-periode, vectorise sur t ;
# (b) methode de tir : bissection sur mu_i pour atteindre Sigma_t x_i = S_i (le cumul est
# strictement decroissant en mu_i) ; (c) Gauss-Seidel sur les agents, car les mu sont
# couples par le prix. Cout LINEAIRE en N, contre O(nbis^N) pour des bissections imbriquees.
NITER_BR = 60

def static_period_N(ceff, kap):
    """ceff [T_,NAG] -> x [T_,NAG]. h_i = 0 donne
       x_i = [alpha - c_i - beta*somme_{j!=i}(1+A_ij)x_j] / (beta*(1+A_ii)), puis clip [0,kappa_i]."""
    x = np.zeros_like(ceff)
    idx = np.arange(NAG)
    for _ in range(NITER_BR):
        for i in range(NAG):
            o = idx[idx != i]
            num = alpha - ceff[:, i] - beta*((1.0 + A_np[i, o])*x[:, o]).sum(1)
            x[:, i] = np.clip(num/(beta*(1.0 + A_np[i, i])), 0.0, kap[i])
    return x

def totals_N(mu, T_, kap, C=None):
    C = C_np if C is None else C
    g = (1.0+r)**np.arange(T_)
    return static_period_N(C[None, :] + mu[None, :]*g[:, None], kap)

def solve_oracle_N(S_, T_, kap=None, nbis=None, nouter=None, C=None):
    kap = KAP_np if kap is None else kap
    nbis = CFG["or_nbis"] if nbis is None else nbis
    nouter = CFG["or_nouter"] if nouter is None else nouter
    mu = np.zeros(NAG); conv = np.inf
    for _ in range(nouter):
        mu_old = mu.copy()
        for i in range(NAG):
            m0 = mu.copy(); m0[i] = 0.0
            if totals_N(m0, T_, kap, C)[:, i].sum() <= S_[i]:   # stock non contraignant
                mu[i] = 0.0; continue
            lo, hi = 0.0, alpha
            for _ in range(nbis):
                m = .5*(lo+hi); mt = mu.copy(); mt[i] = m
                if totals_N(mt, T_, kap, C)[:, i].sum() > S_[i]: lo = m
                else: hi = m
            mu[i] = .5*(lo+hi)
        conv = np.max(np.abs(mu-mu_old))
        if conv < 1e-10: break
    solve_oracle_N.last_conv = float(conv)      # Gauss-Seidel non certifie : on le journalise
    x = totals_N(mu, T_, kap, C)
    return mu, x, alpha - beta*x.sum(1)
solve_oracle_N.last_conv = np.nan

t_or = time.time()
mu_a, xa, pa = solve_oracle_N(S0_np, T, nbis=60, nouter=40)
print(f"STEP 1 oracle ok ({time.time()-t_or:.0f}s) : mu={np.round(mu_a,3)}  "
      f"(convergence Gauss-Seidel {solve_oracle_N.last_conv:.1e})")
for i in range(NAG):
    ai = xa[:, i] > 1e-9
    print(f"  agent {i} (bloc {BLOC[i]:2d}, c={C_np[i]:5.1f}, kap={KAP_np[i]:5.1f}) : "
          f"actif t<={int(np.max(np.nonzero(ai))) if ai.any() else -1}, "
          f"kappa liante {int((xa[:,i]>KAP_np[i]-1e-6).sum())}/{T}, "
          f"cumul {xa[:,i].sum():.1f}/{S0_np[i]:.1f}")
assert np.abs(xa.sum(0) - S0_np).max() < 1e-4, "epuisement viole"
_tlast = int(np.max(np.nonzero(xa.sum(1) > 1e-9)))
assert _tlast < T-1, f"horizon trop court (production encore en t={_tlast})"

# non-regression M2 (N=2 seulement) : l'oracle doit redonner mu=(12.821, 21.757).
if (NAG == 2 and np.allclose(C_np, [10., 20.]) and np.allclose(S0_np, [700., 450.])
        and np.allclose(KAP_np, [20., 30.]) and BLOC[1] < 0):
    assert abs(mu_a[0]-12.821) < 1e-2 and abs(mu_a[1]-21.757) < 1e-2, "oracle N incompatible avec M2"
    print("  [non-regression M2 : mu retrouve]")

SA_T = torch.tensor(S0_np[None, :] - np.concatenate([np.zeros((1, NAG)), np.cumsum(xa, 0)[:-1]], 0))

# regimes, par agent : interieur = ni a zero ni a kappa. C'est la SEULE zone ou lambda_i
# est identifie (ailleurs la FOC est inactive) et la seule ou dx_i/dS_j != 0.
INT = (xa > 1e-6) & (xa < KAP_np[None, :] - 1e-6)
FB_REF = np.array([float(u) for u in CFG["fb_ref"].split(",")])[:NAG]


# STEP 1b : ecran de calibration (cout nul, aucun entrainement)
#   Le feedback markovien vaut somme_{k!=i} coef[i,k]*J[k,j] avec J[k,j] = dx_k/dS_j.
#   J[k,j] est identiquement NUL si l'agent k est a un coin (x_k = 0 ou x_k = kappa_k) :
#   sa decision ne repond alors plus a l'etat. Il faut donc des dates ou DEUX AGENTS
#   STRATEGIQUES sont SIMULTANEMENT INTERIEURS, sinon l'etage C n'a rien a mesurer et
#   l'etage B entraine des objets multiplies par zero.
#   C'est exactement ce qui s'est passe a la calibration M2 : a0 colle a kappa pendant
#   toute la vie de la frange -> J[1,0] = 0 -> feedback = 0 par construction.
#   CRITERE : au moins ~10 dates de recouvrement pour une paire strategique.
def screen_regimes(x, kap=None):
    kap = KAP_np if kap is None else kap
    itr = (x > 1e-6) & (x < kap[None, :] - 1e-6)
    out = {}
    for a_ in range(len(STRAT_IDX)):
        for b_ in range(a_+1, len(STRAT_IDX)):
            i, j = STRAT_IDX[a_], STRAT_IDX[b_]
            out[(i, j)] = int((itr[:, i] & itr[:, j]).sum())
    return itr, out

if CFG["run_screen"]:
    _itr, _ov = screen_regimes(xa)
    print("\nETAPE 1b  ecran de calibration")
    for i in range(NAG):
        print(f"  agent {i} : interieur sur {int(_itr[:,i].sum()):2d} dates "
              f"| kappa {int((xa[:,i]>KAP_np[i]-1e-6).sum()):2d} | zero {int((xa[:,i]<=1e-6).sum()):2d}")
    if _ov:
        for (i, j), n in _ov.items():
            flag = "OK" if n >= 10 else "INSUFFISANT (feedback non identifie)"
            print(f"  recouvrement interieur strategiques ({i},{j}) : {n:2d} dates  -> {flag}")
    else:
        print("  [!] UN SEUL agent strategique : la coincidence CL/OL est un THEOREME")
        print("      (Benchekroun & Withagen 2012). L'etage C n'a rien a mesurer ici.")
        print("      Passer a >= 2 strategiques, p.ex. blocs='0,1,-1'.")


# STEP 1c : grille MPE, induction retrograde N=2 ou 3, independante de PyTorch
#
#   DECISION INTRA-PERIODE : maximisation DIRECTE, sans aucune derivee de V.
#     obj_i(x_i) = (alpha - c_i - beta*somme_{j!=i}(1+A_ij)x_j)*x_i
#                  - (beta(1+A_ii)/2)*x_i^2 + delta*V_i(S - x)
#     d(obj_i)/dx_i = p - beta*Lambda_i - c_i - delta*dV_i/dS_i  =  h_i, EXACTEMENT,
#     et pour toute conduite. Frange : A_ii = A_ij = 0 -> derivee = p - c_f - delta*lam_f,
#     la condition de preneur de prix. C'est un potentiel, pas le profit : c'est ce qui
#     permet de traiter la frange par un argmax alors qu'elle ne maximise pas son profit.
#   Aucun terme de feedback a coder : il est contenu dans le V calcule, puisque
#   sigma_{-i}(S) depend de l'etat.
#
#   PORTE D'ACCEPTATION OBLIGATOIRE : le TEST MONO-AGENT. En mettant tous les stocks
#   rivaux a zero, le jeu disparait, MPE == controle optimal, et l'oracle open-loop est
#   la verite EXACTE. Toute difference est de l'erreur numerique pure. Sans ce test on ne
#   sait pas si un ecart CL/OL est de l'economie ou du bruit.
#   PRECISION ATTEINTE : ~0.5% en 2D a ng=160. Ce plafond vient de l'interpolation
#   multilineaire (pente en escalier). La grille est donc un CONTROLE CROISE, pas le
#   validateur principal : c'est STEP 9 qui valide.
def _interp_ml(Z, coords, grids):
    """Interpolation multilineaire sur grille reguliere. Z de forme (ng,)*N."""
    N = len(grids); ng = Z.shape[0]
    idx, fr = [], []
    for k in range(N):
        d = grids[k][1]
        aa = np.clip(coords[k]/d, 0.0, ng-1-1e-9)
        i0 = aa.astype(np.intp); idx.append(i0); fr.append(aa-i0)
    out = 0.0
    for corner in range(1 << N):
        w = 1.0; ii = []
        for k in range(N):
            b = (corner >> k) & 1
            w = w*(fr[k] if b else (1.0-fr[k]))
            ii.append(idx[k]+b)
        out = out + w*Z[tuple(ii)]
    return out

def mpe_grid(ng=None, nx=None, nbr=None):
    ng = ng or CFG["mpe_ng"]; nx = nx or CFG["mpe_nx"]; nbr = nbr or CFG["mpe_nbr"]
    assert NAG <= 3, "grille MPE : N <= 3 (le cout explose au-dela ; c'est l'argument du PINN)"
    grids = [np.linspace(0.0, S0_np[i], ng) for i in range(NAG)]
    G = np.meshgrid(*grids, indexing="ij")
    shp = G[0].shape
    V = [np.zeros(shp) for _ in range(NAG)]
    XPOL = np.zeros((T, NAG) + shp)
    u = np.linspace(0.0, 1.0, nx); resid = 0.0
    for t in reversed(range(T)):
        if t == T-1:
            # derniere date de decision : aucune continuation, on extrait tout le possible.
            # A ecrire explicitement : interpoler un tableau de zeros n'est pas neutre
            # apres clip aux bords, et cela introduisait une pente parasite.
            x = [np.minimum(KAP_np[i], G[i]) for i in range(NAG)]
        else:
            x = [np.zeros(shp) for _ in range(NAG)]
            for it_br in range(nbr):
                xold = [xx.copy() for xx in x]
                for i in range(NAG):
                    hi = np.minimum(KAP_np[i], G[i])
                    cand = u.reshape((1,)*NAG + (nx,))*hi[..., None]
                    coords = []
                    for k in range(NAG):
                        base = (G[i][..., None] - cand) if k == i else (G[k]-x[k])[..., None]
                        coords.append(np.ascontiguousarray(
                            np.broadcast_to(base, cand.shape)).ravel())
                    cont = _interp_ml(V[i], coords, grids).reshape(cand.shape)
                    lin = alpha - C_np[i] - beta*sum((1.0+A_np[i, k])*x[k]
                                                     for k in range(NAG) if k != i)
                    obj = (lin[..., None]*cand - 0.5*beta*(1.0+A_np[i, i])*cand**2
                           + delta*cont)
                    x[i] = np.take_along_axis(cand, obj.argmax(-1)[..., None], -1)[..., 0]
                if it_br == nbr-1:
                    resid = max(resid, max(float(np.abs(x[i]-xold[i]).max()) for i in range(NAG)))
        p = alpha - beta*sum(x)
        Sp = [np.ascontiguousarray(G[i]-x[i]).ravel() for i in range(NAG)]
        if t == T-1:
            V = [(p - C_np[i])*x[i] for i in range(NAG)]
        else:
            V = [(p - C_np[i])*x[i] + delta*_interp_ml(V[i], Sp, grids).reshape(shp)
                 for i in range(NAG)]
        for i in range(NAG): XPOL[t, i] = x[i]
    return grids, XPOL, resid

def mpe_rollout(grids, XPOL, S_init=None):
    """Deroule la politique MPE de la grille depuis S_init. -> X [T,NAG], P [T]."""
    S = np.array(S_init if S_init is not None else S0_np, dtype=float)
    X, P = [], []
    for t in range(T):
        co = [np.array([S[k]]) for k in range(NAG)]
        xt = np.array([float(_interp_ml(XPOL[t, i], co, grids)[0]) for i in range(NAG)])
        xt = np.clip(np.minimum(xt, S), 0.0, None)
        X.append(xt.copy()); P.append(alpha - beta*xt.sum()); S = S - xt
    return np.array(X), np.array(P)

# masque des dates d'extinction (+/-1), reutilise en STEP 5.
_ext = np.zeros(T, dtype=bool)
for i in range(NAG):
    _mi = np.nonzero(xa[:, i] > 1e-9)[0]
    if _mi.size: _ext[max(0, _mi[-1]-1):_mi[-1]+2] = True

MPE = {}
Xcl = Pcl = None
if CFG["run_mpe_grid"] and NAG <= 3:
    _tg = time.time()
    _grids, _XPOL, _rs = mpe_grid()
    # PORTE : verite mono-agent. Rivaux a stock nul -> plus de jeu -> l'oracle est exact.
    _Smono = S0_np.copy()
    for k in range(1, NAG): _Smono[k] = 0.0
    _, _xmo, _pmo = solve_oracle_N(_Smono, T, nbis=60, nouter=40)
    _amo = _xmo[:, 0] > 1e-9
    Xmo, Pmo = mpe_rollout(_grids, _XPOL, _Smono)
    _emp = float(np.abs(Pmo[_amo]-_pmo[_amo]).max()/_pmo.max())
    Xcl, Pcl = mpe_rollout(_grids, _XPOL)
    _a = xa.sum(1) > 1e-6; _keepg = _a & (~_ext)
    _e = np.abs(Pcl[_a]-pa[_a])/pa.max()
    MPE = dict(ng=CFG["mpe_ng"], br_resid=_rs, mono_err_p=_emp,
               gap_p=float(_e.max()),
               gap_p_noext=float(np.abs(Pcl[_keepg]-pa[_keepg]).max()/pa.max()),
               gap_p_q90=float(np.quantile(_e, .90)),
               gap_p0=float(abs(Pcl[0]-pa[0])/pa[0]), cum=Xcl.sum(0).tolist())
    print(f"\nETAPE 2  GRILLE MPE ({time.time()-_tg:.0f}s, ng={CFG['mpe_ng']}, nx={CFG['mpe_nx']})")
    print(f"  PORTE mono-agent (erreur numerique pure) : {_emp:.2%} "
          f"{'OK' if _emp < 5e-3 else '-> plancher de la grille, tout ecart CL/OL en dessous est du bruit'}")
    print(f"  residu BR intra-periode : {_rs:.3f} bbl")
    print(f"  epuisement CL {np.round(Xcl.sum(0),1)} / {S0_np}")
    print("   t  | " + " | ".join(f"a{i} CL/OL" for i in range(NAG)) + " |  p CL/OL")
    for tt in [0, T//5, 2*T//5, 3*T//5, _tlast]:
        s = " | ".join(f"{Xcl[tt,i]:5.1f}/{xa[tt,i]:5.1f}" for i in range(NAG))
        print(f"  {tt:3d} | {s} | {Pcl[tt]:6.2f}/{pa[tt]:6.2f}")
    print(f"  ECART CL/OL : max {MPE['gap_p']:.2%} | hors ext {MPE['gap_p_noext']:.2%} "
          f"| q90 {MPE['gap_p_q90']:.2%} | t=0 {MPE['gap_p0']:.2%}")
    print(f"  A COMPARER au plancher mono-agent {_emp:.2%}. En dessous, ce n'est pas de l'economie.")


# STEP 2 : reseaux
# un reseau par agent, trois tetes (politique z, valeur w, co-etat l), tronc partage.
# Chaque reseau prend l'etat COMPLET S : c'est la definition du Markov, et c'est ce qui rend
# les derivees croisees dV_i/dS_j et dsigma_k/dS_j calculables par autodiff.
# NOTE DAG : le nombre de reseaux VALEUR utiles = nombre d'agents STRATEGIQUES, pas N.
# La tete valeur de la frange n'est lue par aucune equation une fois STRAT applique ; sa
# tete POLITIQUE, elle, est indispensable (J[f,.] entre dans le feedback des strategiques).
# - ancrage terminal EXACT par 1{t<T}
# - SiLU : le residu Sobolev a besoin de dV/dS non trivial ; le kink est deja dans
#   l'architecture via les min(), donc rien de non lisse a apprendre.
class Net(nn.Module):
    def __init__(s, width, emb_dim, depth):
        super().__init__()
        s.emb = nn.Embedding(T+1, emb_dim)          # T+1 : l'index t=T existe, neutralise par alive
        nn.init.normal_(s.emb.weight, std=0.1)
        layers, d = [], NAG + emb_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]; d = width
        s.trunk = nn.Sequential(*layers)
        s.head_z = nn.Linear(width, 1); s.head_w = nn.Linear(width, 1); s.head_l = nn.Linear(width, 1)
    def forward(s, S, ti):
        h = s.trunk(torch.cat([S/S0V, s.emb(ti)], dim=-1))
        return s.head_z(h).squeeze(-1), s.head_w(h).squeeze(-1), s.head_l(h).squeeze(-1)

NETS = nn.ModuleList([Net(CFG["width"], CFG["emb_dim"], CFG["depth"]) for _ in range(NAG)])
PARAMS = list(NETS.parameters())

def forward_all(S, ti, snap=False):
    """S:[B,NAG] ti:[B] long -> x, V, lam tous [B,NAG]"""
    z, w, l = [], [], []
    for i in range(NAG):
        zi, wi, li = NETS[i](S, ti); z.append(zi); w.append(wi); l.append(li)
    z = torch.stack(z, -1); w = torch.stack(w, -1); l = torch.stack(l, -1)
    # x = min(min(S,kappa), softplus(z)) : atteint kappa EXACTEMENT (coin superieur franc),
    # garantit 0 <= x <= S par construction. La faisabilite n'est jamais a apprendre.
    x = torch.minimum(torch.minimum(S, KAP), F.softplus(z))
    if snap:                                        # EVAL uniquement (zone morte a l'entrainement)
        x = torch.where(x < SNAP_EPS, torch.zeros_like(x), x)
    alive = (ti < T).double().unsqueeze(-1)
    G = growth_t(ti).unsqueeze(-1) if CFG["G_in_V"] else torch.ones_like(alive)
    # V libre. Les deux ancrages testes sont morts et ne doivent pas revenir :
    #   * (S/S0V) : annule V_i et TOUS ses gradients sur l'hyperplan S_i=0, donc detruit
    #     l'ancre terminale de la chaine arriere de R_sob/R_soff.
    #   * (* S)   : impose V_i = lambda_i*S_i, i.e. courbure nulle en S, alors que la
    #     courbure est l'objet meme que l'etage B doit apprendre (mesure : facteur 1.7).
    V   = alive*F.softplus(w) * VREF*G
    lam = alive*F.softplus(l)*LREF*G
    return x, V, lam


# STEP 2b : outils pathwise (V_MC, cible du gradient)
# Ils ne comparent le reseau qu'a ce qu'il produit lui-meme en deroulant sa propre
# politique jusqu'a T. Aucune cible bootstrap, aucun oracle. C'est ce qui les rend
# utilisables a l'etage C, ou la surface de valeur bascule de open-loop vers closed-loop.
def mc_value(Sb, tb):
    """Sb [M,NAG], tb [M] -> V^MC [M,NAG] sous la politique COURANTE."""
    with torch.no_grad():
        S = Sb.clone(); XS, PS = [], []
        for t in range(T):
            live = ((tb + t) < T).double().unsqueeze(-1)
            x, _, _ = forward_all(S, torch.clamp(tb + t, max=T), snap=True)
            x = x*live
            XS.append(x.clone()); PS.append(alpha - beta*x.sum(-1, keepdim=True))
            S = S - x
        Vmc = torch.zeros(Sb.shape[0], NAG)
        for t in reversed(range(T)):
            Vmc = (PS[t] - COST)*XS[t] + delta*Vmc
    return Vmc

def gv_target_fd(S0b, t0b, dlt=None, freeze=False):
    """S0b [M,NAG], t0b [M] -> [M,NAG,NAG] cible de dV_i/dS_j, difference CENTREE."""
    dlt = (CFG["fd_rel"]*S0V) if dlt is None else dlt
    M = S0b.shape[0]
    blocks = [S0b]
    for j in range(NAG):
        ej = torch.zeros(NAG); ej[j] = 1.0
        blocks += [S0b + ej*dlt, torch.clamp(S0b - ej*dlt, min=0.0)]
    base = torch.cat(blocks, 0)
    tb = t0b.repeat(2*NAG+1)
    if not freeze:
        Vmc = mc_value(base, tb)
    else:
        # rivaux GELES sur la trajectoire NON perturbee : chaque colonne j a son propre
        # masque, d'ou la boucle explicite (non vectorisable en une expression).
        with torch.no_grad():
            Sref = S0b.clone(); XR = []
            for t in range(T):
                live = ((t0b + t) < T).double().unsqueeze(-1)
                xr_, _, _ = forward_all(Sref, torch.clamp(t0b + t, max=T), snap=True)
                xr_ = xr_*live; XR.append(xr_.clone()); Sref = Sref - xr_
            Vmc = torch.zeros(base.shape[0], NAG)
            for j in range(NAG):
                for sgn, off in ((+1, 1+2*j), (-1, 2+2*j)):
                    S = base[off*M:(off+1)*M].clone()
                    XS, PS = [], []
                    for t in range(T):
                        live = ((t0b + t) < T).double().unsqueeze(-1)
                        xp, _, _ = forward_all(S, torch.clamp(t0b + t, max=T), snap=True)
                        x = xp*live
                        for k in range(NAG):
                            if k != j: x[:, k] = XR[t][:, k]
                        x = torch.minimum(x, S)
                        XS.append(x.clone()); PS.append(alpha - beta*x.sum(-1, keepdim=True))
                        S = S - x
                    v = torch.zeros(M, NAG)
                    for t in reversed(range(T)):
                        v = (PS[t] - COST)*XS[t] + delta*v
                    Vmc[off*M:(off+1)*M] = v
            Vmc[:M] = mc_value(base[:M], t0b)
    out = torch.zeros(M, NAG, NAG)
    for j in range(NAG):
        out[:, :, j] = (Vmc[(1+2*j)*M:(2+2*j)*M] - Vmc[(2+2*j)*M:(3+2*j)*M])/(2*dlt[j])
    return out

_GS = _GTT = _GTGT = None
def refresh_gv_target(m=None):
    """Cible pathwise du gradient, sur un sous-echantillon du buffer. Analogue de _VBUF."""
    global _GS, _GTT, _GTGT
    m = m or CFG["gv_pts"]
    idx = torch.randint(0, _BUF[0].shape[0], (m,))
    _GS, _GTT = _BUF[0][idx].clone(), _BUF[1][idx].clone()
    _GTGT = gv_target_fd(_GS, _GTT).detach()

def gv_reg_loss(nb=None):
    """Regression de gV sur la cible MESUREE. Terme SEPARE : ne touche ni residuals ni
       sample_states. Masque STRAT sur les lignes (V de la frange n'est lue par rien)."""
    nb = nb or CFG["gv_batch"]
    k = torch.randint(0, _GS.shape[0], (min(nb, _GS.shape[0]),))
    Sg_ = _GS[k].clone().requires_grad_(True); tg_ = _GTT[k]
    _, Vg_, _ = forward_all(Sg_, tg_)
    g_ = torch.stack([torch.autograd.grad(Vg_[:, i].sum(), Sg_, create_graph=True)[0]
                      for i in range(NAG)], dim=1)
    Gt_ = growth_t(tg_).view(-1, 1, 1)
    w_ = STRAT.view(1, -1, 1)
    return (((g_ - _GTGT[k])/(LREF*Gt_)*w_)**2).sum((-1, -2)).mean()


# STEP 3 : residus et loss
# FOC -> politique ; enveloppe -> co-etat ; Bellman -> valeur ; Sobolev -> lie les deux.
# Chaque agent n'a que SA PROPRE FOC. Ce qui fabrique l'equilibre : p n'est PAS detache
# (le gradient du residu de i remonte dans les reseaux de tous les autres) et chaque reseau
# lit l'etat complet. Tout le reste est de la regression supervisee sur cible bootstrap
# detachee : schema semi-gradient.
# R_foc est EXACT et non approche : h_i etant affine en x_i de pente -beta(1+A_ii), le pas
# x + h/DEN atterrit sur la racine exacte, et le clamp projette. R_foc = 0 <=> x est la
# meilleure reponse contrainte exacte etant donne (p, Lambda, lambda').
def residuals(S, ti):
    S = S.detach().requires_grad_(True)
    x, V, lam = forward_all(S, ti)
    p = alpha - beta*x.sum(-1, keepdim=True)
    Lam = x @ A.T
    S2 = S - x
    _, V2, lam2 = forward_all(S2, ti+1)
    lam2d = lam2.detach()
    cap = torch.minimum(S, KAP)

    h = p - beta*Lam - COST - delta*lam2d
    R_foc = x - torch.clamp(x + h/DEN, torch.zeros_like(x), cap)

    gV = torch.stack([torch.autograd.grad(V[:, i].sum(), S, create_graph=True)[0]
                      for i in range(NAG)], dim=1)

    # Multiplier par 0.0 preserve le lien au graphe d'autodifferenciation
    feedback = x * 0.0
    R_soff = gV * 0.0
    R_sob = x * 0.0

    if CFG["sob_off"]:
        J   = torch.stack([torch.autograd.grad(x[:, k].sum(), S, create_graph=True)[0]
                           for k in range(NAG)], dim=1)

        GV2 = torch.stack([torch.autograd.grad(V2[:, i].sum(), S2, create_graph=True)[0]
                           for i in range(NAG)], dim=1)
        # coef[i,k] = d(pi_i)/d(x_k) - delta*dV_i'/dS_k'
        #           = -beta*x_i - delta*GV2[i,k]  (k!=i)   |   + (p - c_i)  (k=i)
        coef = -beta*x.unsqueeze(-1) - delta*GV2 + torch.diag_embed(p - COST)

        # M_full  : dV_i/dS_j = delta*GV2[i,j] + M_full[i,j]. Somme sur TOUS les k.
        # M_cross : somme sur k != i seulement. Le terme diagonal coef_ii*J_ii n'est PAS du
        #           feedback, c'est le coin de conduite.
        M_full  = torch.einsum('bik,bkj->bij', coef,     J)
        M_cross = torch.einsum('bik,bkj->bij', coef*OFF, J)

        feedback = torch.diagonal(M_cross, dim1=1, dim2=2)
        R_soff   = (gV - (delta*GV2 + M_full).detach())*OFF
        # ATTENTION : cette cible est AUTO-REFERENTIELLE (construite depuis le V du
        # reseau lui-meme, ancre terminale degeneree a 0). R_sob peut converger vers un
        # point fixe auto-coherent mais faux la ou il est le seul residu actif. C'est la
        # raison de feedback_only. Alternative mesuree : gv_target_fd (9f).
        R_sob = torch.diagonal(gV, dim1=1, dim2=2) - (delta*torch.diagonal(GV2, dim1=1, dim2=2)
                                            + torch.diagonal(M_full, dim1=1, dim2=2)).detach()

        # Une frange PRENEUSE DE PRIX n'a pas de terme de feedback. Une fois feedback_f = 0,
        # GV2[f,:] n'entre plus dans aucune equation.
        R_soff   = R_soff * STRAT.view(1, -1, 1)
        feedback = feedback * STRAT

        if CFG["soff_mode"] != "bellman":
            # "off" et "pathwise" : on garde J, GV2 et feedback, on annule les residus
            # Sobolev construits depuis le V du reseau. En mode "pathwise" la supervision
            # du gradient est assuree par un terme SEPARE, sur cible mesuree (STEP 4).
            R_sob = R_sob*0.0; R_soff = R_soff*0.0

    bind = (x >= KAP - 1e-4).detach()
    zero = (x <= SNAP_EPS).detach()
    marg = p.detach()-beta*Lam.detach()-COST        # marge strategique, PAS encore clampee
    if CFG["env_hard"]:
        # regime INTERIEUR -> cible DURE mesuree, sans bootstrap.
        # regimes kappa et zero -> enveloppe lambda = delta*lambda'.
        tgt = torch.where(bind | zero, delta*lam2d, marg)
    else:
        tgt = torch.where(bind, delta*lam2d, torch.maximum(marg, delta*lam2d))
    # Injection du MPE : UNE ligne. Le feedback entre dans la cible du CO-ETAT, puis se
    # propage a la politique via le terme delta*lambda' de la FOC. Correct dans les deux
    # branches : en zone kappa J_ii = 0, a l'interieur coef_ii = 0 pour un singleton, donc
    # dV_i/dS_i = delta*lambda'_i + feedback_i dans les deux cas.
    if CFG["cross"]: tgt = tgt + CFG["fb_gain"]*feedback.detach()
    # CLAMP APRES LE FEEDBACK, sur la cible TOTALE. lam = softplus(.)*LREF*G est
    # STRICTEMENT positif : une cible negative est inatteignable, R_env reste positif en
    # permanence et pousse softplus -> 0, un point fixe faux. Le feedback est negatif sur
    # 37% des points non nuls, donc ce cas se produit. Clamper `marg` seul ne couvrait pas
    # la branche kappa/zero, ou delta*lambda' + feedback peut aussi passer sous zero.
    # A cross=False ce clamp est strictement equivalent au precedent (delta*lam2d >= 0).
    tgt = torch.clamp(tgt, min=0.0)
    R_env = lam - tgt

    R_bell = V - ((p.detach()-COST)*x.detach() + delta*V2.detach())

    R_lam = (torch.diagonal(gV, dim1=1, dim2=2) - lam) * SOB_OK

    # R_dfoc : Sobolev sur la FOC. Ecrit comme RESIDU sur le J verifie (4.7e-5 contre les
    # differences finies) et NON comme substitution : substituer echangerait J contre
    # L = dlambda'/dS', que rien ne supervise. lam2 est ici NON detache, ce qui contraint
    # conjointement J et L -- c'est la force du residu et sa faiblesse (cf. CFG).
    # Masque free : dh_i/dS_j = 0 n'est vraie que la ou la FOC est ACTIVE ; aux coins
    # h_i != 0 et sa derivee n'est pas contrainte. free est deja calcule pour env_hard.
    R_dfoc = torch.zeros(S.shape[0], NAG, NAG, dtype=S.dtype)
    if CFG["dfoc"]:
        h_full = p - beta*Lam - COST - delta*lam2
        dh = torch.stack([torch.autograd.grad(h_full[:, i].sum(), S, create_graph=True)[0]
                          for i in range(NAG)], dim=1)          # [B, NAG, NAG]
        free = (~(bind | zero)).to(S.dtype)
        R_dfoc = dh * free.unsqueeze(-1) * STRAT.view(1, -1, 1)

    return R_foc, R_env, R_bell, R_sob, R_soff, R_lam, V, R_dfoc

def loss_from_res(R, ti, Vtgt=None, msk=None, parts=False):
    # UNE loss, somme sur tous les agents, un seul optimiseur conjoint.
    R_foc, R_env, R_bell, R_sob, R_soff, R_lam, V, R_dfoc = R
    G_t = growth_t(ti).unsqueeze(-1)
    GB = G_t if CFG["norm_G_bell"] else torch.ones_like(G_t)
    GL = G_t if CFG["norm_G_lam"]  else torch.ones_like(G_t)
    c_foc  =        ((R_foc /XREF      )**2).sum(-1)
    c_env  =        ((R_env /(LREF*GL) )**2).sum(-1)
    c_bell =        ((R_bell/(VREF*GB) )**2).sum(-1)
    c_sob  = W_SOB *((R_sob /(LREF*GL) )**2).sum(-1)
    c_lam  = W_SOB *((R_lam /(LREF*GL) )**2).sum(-1)
    L = c_foc + c_env + c_bell + c_sob + c_lam
    c_soff = torch.zeros_like(L)
    if CFG["sob_off"]:
        c_soff = W_SOFF*((R_soff/(LREF*GL.unsqueeze(-1)))**2).sum((-1, -2))
        L = L + c_soff

    # echelle naturelle de dh/dS : une derivee prix/stock, soit LREF/S0. La division porte
    # sur le dernier axe (j), celui de la variable de derivation.
    c_dfoc = torch.zeros_like(L)
    if CFG["dfoc"]:
        c_dfoc = W_DFOC*((R_dfoc/(LREF/S0V))**2).sum((-1, -2))
        L = L + c_dfoc

    # Cible Monte-Carlo TD(1) pour V, sur le tube. R_bell est un residu a UN pas : il peut
    # etre minuscule pendant que V derive de 25% (V est un bootstrap a 20-40 pas avec une
    # unique ancre en T). Le buffer fait deja 64 rollouts complets : la cible est gratuite,
    # exacte, sans oracle, et valide a l'etage C (V^MC est la valeur de la politique
    # COURANTE, quelle qu'elle soit). msk = 1 uniquement sur le bras tube pur.
    # ATTENTION : STEP 6 compare V_net(S0,0) au profit du rollout depuis S0, soit
    # exactement l'objet regresse ici -> Bellman MC devient QUASI-TAUTOLOGIQUE.
    c_vmc = torch.zeros_like(L)
    if CFG["use_vmc"] and (Vtgt is not None) and (msk is not None):
        c_vmc = CFG["w_vmc"]*(((V - Vtgt.detach())/(VREF*GB))**2).sum(-1)*msk
        L = L + c_vmc

    if parts:
        d = dict(foc=c_foc, env=c_env, bell=c_bell, sob=c_sob, lam=c_lam,
                 soff=c_soff, vmc=c_vmc, dfoc=c_dfoc)
        return L.mean(), {k: v.mean().item() for k, v in d.items()}
    return L.mean()

# normes de gradient par residu. Sans ces chiffres on pilote a l'aveugle.
def grad_norms(Sb, tb, Vtgt=None, msk=None):
    R = residuals(Sb, tb)
    G_t = growth_t(tb).unsqueeze(-1)
    GB = G_t if CFG["norm_G_bell"] else torch.ones_like(G_t)
    GL = G_t if CFG["norm_G_lam"]  else torch.ones_like(G_t)
    comps = dict(
        foc  =        ((R[0]/XREF     )**2).sum(-1).mean(),
        env  =        ((R[1]/(LREF*GL))**2).sum(-1).mean(),
        bell =        ((R[2]/(VREF*GB))**2).sum(-1).mean(),
        sob  = W_SOB *((R[3]/(LREF*GL))**2).sum(-1).mean(),
        lam  = W_SOB *((R[5]/(LREF*GL))**2).sum(-1).mean(),
    )
    if CFG["sob_off"]:
        comps["soff"] = (W_SOFF*((R[4]/(LREF*GL.unsqueeze(-1)))**2).sum((-1, -2))).mean()
    if CFG["dfoc"]:
        comps["dfoc"] = (W_DFOC*((R[7]/(LREF/S0V))**2).sum((-1, -2))).mean()
    if CFG["use_vmc"] and (Vtgt is not None) and (msk is not None):
        comps["vmc"] = (CFG["w_vmc"]*(((R[6]-Vtgt.detach())/(VREF*GB))**2).sum(-1)*msk).mean()
    out = {}
    for k, v in comps.items():
        if float(v) == 0.0: out[k] = 0.0; continue
        g = torch.autograd.grad(v, PARAMS, retain_graph=True, allow_unused=True)
        out[k] = float(torch.sqrt(sum((gi**2).sum() for gi in g if gi is not None)))
    return out


# STEP 4a : buffer et sampler
# tube (buffer de la politique COURANTE : voisinage transverse, force de rappel du rollout)
# + petit ancrage uniforme (structure globale). Le tube-oracle a bruit nul ne donnait aucun
# champ transverse => loss 1e-6 et trajectoire fausse a 16%.
# LA LOSS NE VALIDE RIEN : une loss 65x pire a battu l'autre d'un facteur 2.5 sur le rollout.
# A N grand l'ancrage uniforme couvre de moins en moins (volume exponentiel) : c'est le TUBE
# qui rend le DEQN faisable la ou une grille ne l'est pas.
_BUF = None
_VBUF = None
def refresh_buffer(n=None):
    global _BUF, _VBUF
    n = n or CFG["buf_paths"]
    with torch.no_grad():
        S = (CFG["buf_lo"] + (CFG["buf_hi"]-CFG["buf_lo"])*torch.rand(n, NAG))*S0V
        SS, TT, XX, PP = [], [], [], []
        for t in range(T):
            ti = torch.full((n,), t, dtype=torch.long)
            SS.append(S.clone()); TT.append(ti)
            x, _, _ = forward_all(S, ti)
            XX.append(x.clone()); PP.append(alpha - beta*x.sum(-1, keepdim=True))
            S = S - x
        _BUF = (torch.cat(SS, 0), torch.cat(TT, 0), n)
        # accumulation ARRIERE du profit actualise le long du rollout DEJA calcule.
        # V^MC_i(S_t,t) = somme_{s>=t} delta^(s-t) (p_s - c_i) x_is , tronquee en T.
        Vmc, VV = torch.zeros(n, NAG), [None]*T
        for t in reversed(range(T)):
            Vmc = (PP[t] - COST)*XX[t] + delta*Vmc
            VV[t] = Vmc.clone()
        _VBUF = torch.cat(VV, 0)

def sample_states(B):
    if CFG["sampler"] == "tube_oracle":
        idx = torch.randint(0, T, (B,))
        return SA_T[idx], idx, torch.zeros(B, NAG), torch.zeros(B)

    n_u = int(CFG["frac_unif"] * B)
    n_j = int(CFG["frac_gauss"] * B)
    n_d = int(CFG.get("frac_decor", 0.0) * B)
    n_b = B - n_u - n_j - n_d

    S_u = torch.rand(n_u, NAG) * S0V * BOX                      # 1. global uniforme
    t_u = torch.randint(0, T, (n_u,))

    idx_j = torch.randint(0, _BUF[0].shape[0], (n_j,))          # 2. jitter local
    S_j = torch.clamp(_BUF[0][idx_j] + torch.randn(n_j, NAG) * 0.05 * S0V, min=0.0)
    t_j = _BUF[1][idx_j]

    idx_b = torch.randint(0, _BUF[0].shape[0], (n_b,))          # 3. tube pur
    S_b = _BUF[0][idx_b]; t_b = _BUF[1][idx_b]

    # 4. bras decorrelant (brise la collinearite S_0 <-> t). Le confondant est ETABLI mais
    # ce bras ne le casse pas : la cible de gV[i,j] est construite depuis le V du reseau
    # lui-meme. Point fixe auto-referentiel, qu'aucun echantillonnage ne casse.
    # -> frac_decor=0.0 par defaut ; le bras reste inerte mais present.
    idx_d = torch.randint(0, _BUF[0].shape[0], (n_d,))
    S_d = _BUF[0][idx_d].clone()
    agent_idx = torch.randint(0, NAG, (n_d,))
    if n_d: S_d[torch.arange(n_d), agent_idx] = torch.rand(n_d) * S0V[agent_idx] * BOX
    t_d = _BUF[1][idx_d]

    # la cible MC n'est VALIDE QUE sur le bras 3 (tube pur, etat non perturbe).
    Vt = torch.cat([torch.zeros(n_u, NAG), torch.zeros(n_j, NAG),
                    (_VBUF[idx_b] if _VBUF is not None else torch.zeros(n_b, NAG)),
                    torch.zeros(n_d, NAG)], 0)
    mk = torch.cat([torch.zeros(n_u), torch.zeros(n_j),
                    torch.ones(n_b),  torch.zeros(n_d)], 0)
    return (torch.cat([S_u, S_j, S_b, S_d], 0), torch.cat([t_u, t_j, t_b, t_d], 0), Vt, mk)


# STEP 4b : entrainement Adam (+ L-BFGS)
print(f"\nSTEP 4 : Adam (NAG={NAG}, cross={CFG['cross']}, sob_off={CFG['sob_off']}, "
      f"vmc={CFG['use_vmc']})...")
_PATHW = CFG["sob_off"] and CFG["soff_mode"] == "pathwise"
opt = torch.optim.Adam(PARAMS, lr=CFG["lr"])
T_START = time.time(); refresh_buffer(); loss_hist = []
CKPT = f"ckpt_{CFG['tag']}.pt"
_ITERS = CFG["iters"]
if CFG["load_ckpt"] and os.path.exists(CKPT):
    NETS.load_state_dict(torch.load(CKPT)); _ITERS = 0
    print(f"  reseau RECHARGE depuis {CKPT} : entrainement saute.")
if _PATHW:
    print(f"  supervision du gradient : PATHWISE ({CFG['gv_pts']} pts, "
          f"rafraichie tous les {CFG['gv_every']} pas, batch {CFG['gv_batch']})")
for it in range(_ITERS):
    if it == CFG["iters"]*4//10:
        for g in opt.param_groups: g["lr"] = 3e-4
    if it == CFG["iters"]*6//10:
        for g in opt.param_groups: g["lr"] = 1e-4
    if CFG["sampler"] == "mix" and it % CFG["buf_every"] == 0: refresh_buffer()
    opt.zero_grad()
    Sb, tb, Vb, mb = sample_states(CFG["batch"])
    L = loss_from_res(residuals(Sb, tb), tb, Vb, mb)
    if _PATHW and (it % CFG["gv_every"] == 0 or _GS is None): refresh_gv_target()
    if _PATHW: L = L + W_SOFF*gv_reg_loss()
    L.backward(); opt.step()
    if it % 100 == 0: loss_hist.append((it, L.item()))
    if it % 3000 == 0: print(f"  it={it:5d} loss={L.item():.3e} ({time.time()-T_START:.0f}s)", flush=True)
    if CFG["diag_every"] and it % CFG["diag_every"] == 0:
        Sd, td, Vd, md = sample_states(CFG["batch"])
        _, pc = loss_from_res(residuals(Sd, td), td, Vd, md, parts=True)
        gn = grad_norms(Sd, td, Vd, md)
        print("        parts " + " ".join(f"{k}={v:.2e}" for k, v in pc.items()), flush=True)
        print("        |grad| " + " ".join(f"{k}={v:.2e}" for k, v in gn.items()), flush=True)

if _ITERS:
    torch.save(NETS.state_dict(), CKPT); print(f"  reseau sauve -> {CKPT}")
print("STEP 4 : L-BFGS (collocation fixe)...")
Lpost = loss_hist[-1][1] if loss_hist else float("nan")
for outer in range(CFG["lbfgs_outer"]):
    if CFG["sampler"] == "mix": refresh_buffer(256)
    Sb, tb, Vb, mb = sample_states(6000)
    Sb = torch.cat([Sb, S0V.expand(100, NAG)], 0)   # ancre S0 dans le jeu fixe
    tb = torch.cat([tb, torch.zeros(100, dtype=torch.long)], 0)
    Vb = torch.cat([Vb, torch.zeros(100, NAG)], 0)
    mb = torch.cat([mb, torch.zeros(100)], 0)
    if _PATHW: refresh_gv_target()
    opt2 = torch.optim.LBFGS(PARAMS, lr=1.0, max_iter=400, history_size=60,
                             tolerance_grad=1e-15, tolerance_change=1e-16,
                             line_search_fn="strong_wolfe")
    def closure():
        opt2.zero_grad(); Lf = loss_from_res(residuals(Sb, tb), tb, Vb, mb)
        if _PATHW: Lf = Lf + W_SOFF*gv_reg_loss()
        Lf.backward(); return Lf
    L = opt2.step(closure); Lpost = closure().item(); loss_hist.append((CFG["iters"]+outer, Lpost))
    print(f"  outer={outer} loss_pre={L.item():.3e} loss_post={Lpost:.3e} ({time.time()-T_START:.0f}s)", flush=True)
LOSS_FINAL = Lpost


# STEP 5 : rollout vs oracle
def rollout(S_init, snap=True):
    S = S_init.clone().reshape(1, NAG); X, P = [], []
    with torch.no_grad():
        for t in range(T):
            x, _, _ = forward_all(S, torch.tensor([t]), snap=snap)
            X.append(x[0].clone().numpy()); P.append((alpha - beta*x.sum()).item())
            S = S - x
    return np.array(X), np.array(P), S[0].numpy()

X, P, Send = rollout(S0V)
act = xa.sum(1) > 1e-6
err_p = np.abs(P[act]-pa[act]).max()/pa.max()
err_x = np.abs(X[act]-xa[act]).max()/KAP_np.max()
print(f"\nSTEP 5 rollout vs oracle open-loop "
      f"({'COMPARATEUR (MPE)' if CFG['cross'] else 'VALIDATEUR'})")
print("  t  | " + " | ".join(f"a{i} res/or" for i in range(NAG)) + " |  p res/or")
for tt in [0, T//5, 2*T//5, 3*T//5, _tlast]:
    s = " | ".join(f"{X[tt,i]:5.1f}/{xa[tt,i]:5.1f}" for i in range(NAG))
    print(f" {tt:3d} | {s} | {P[tt]:6.2f}/{pa[tt]:6.2f}")
print(f"  epuisement {np.round(X.sum(0),1)} / {S0_np}")
print(f"  err_max(actives) x={err_x:.2e}  p={err_p:.2e}")

ep = np.abs(P[act]-pa[act])/pa.max(); ia = np.nonzero(act)[0]
top = np.argsort(ep)[::-1][:4]
print("  4 pires dates :", ", ".join(f"t={ia[i]}({ep[i]*100:.1f}%)" for i in top))

# METRIQUES ROBUSTES : err_p est un MAX, donc pilote par la seule date d'extinction, ou
# le biais structurel de date d'extinction domine. err_p_noext exclut les dates
# d'extinction +/-1 ; err_p_q90 est insensible a un point isole.
# ATTENTION A L'ESTIMAND : la "cible M2 de 9e-3" etait un err_p COMPLET. La comparer a
# err_p_noext flatte mecaniquement le run.
_keep = act & (~_ext)
ERRP_NOEXT = float(np.abs(P[_keep]-pa[_keep]).max()/pa.max()) if _keep.any() else float("nan")
ERRP_Q90 = float(np.quantile(ep, 0.90))
print(f"  err_p hors dates d'extinction = {ERRP_NOEXT:.2e}   |   err_p q90 = {ERRP_Q90:.2e}")

# Hotelling : marge STRATEGIQUE h_i = p - beta*Lambda_i - c_i, deflatee -> doit etre PLATE
# a la hauteur mu_i. La PLATITUDE (cv) est un test SANS ORACLE ; le NIVEAU utilise mu.
# Ne porte que sur les dates INTERIEURES, seules ou lambda est identifie.
# A cross=True la courbe n'est plus plate, et l'ecart a la platitude EST le feedback.
Lam_r = X @ A_np.T
hall = (P[:, None] - beta*Lam_r - C_np[None, :])/((1.0+r)**np.arange(T))[:, None]
HOT = {}
for i in range(NAG):
    mi = (X[:, i] > 1e-3) & (X[:, i] < KAP_np[i]-1e-3)
    if mi.sum() > 1:
        HOT[i] = (hall[mi, i].std()/abs(hall[mi, i].mean()), hall[mi, i].mean(),
                  int(mi.sum()))
        print(f"  Hotelling a{i} : cv={HOT[i][0]:.2e} sur {int(mi.sum())} dates | "
              f"niveau {HOT[i][1]:.2f} vs mu={mu_a[i]:.2f}")


# STEP 5a : rente par date
lam_true = mu_a[None, :]*((1.0+r)**np.arange(T))[:, None]
with torch.no_grad():
    _, _, lam_net = forward_all(SA_T, torch.arange(T))
rel_lam = np.abs(lam_net.numpy()-lam_true)/np.maximum(lam_true, 1e-9)
act_nag = xa > 1e-6      # masque 2D : dates actives de CHAQUE agent (pas un masque global)
LAM_MED = float(np.median(rel_lam[act_nag]))
LAM_MAX = float(rel_lam[act_nag].max())
print(f"\nSTEP 5a rente le long du sentier : err rel mediane {LAM_MED:.2%}, max {LAM_MAX:.2%}")
for i in range(NAG):
    m = xa[:, i] > 1e-6
    if m.any():
        print(f"  a{i}: med {np.median(rel_lam[m, i]):.2%} max {rel_lam[m, i].max():.2%}")


# STEP 5e : rente par ses deux representations
# 5a mesure la tete lambda. Ici dV_i/dS_i, meme verite, meme echantillon.
# ATTENTION AUX AGENTS NON SINGLETONS (SOB_OK=0) : pour la frange, dV_f/dS_f n'est PAS
# lambda_f. L'ecart structurel est le COIN DE CONDUITE beta*x_f*J_ff/lambda_f, MESURE a
# ~13%. 5e compare donc deux objets DIFFERENTS pour un non-singleton : c'est la metrique
# qui est mal specifiee, pas le reseau. Pour la frange, lire 9f, pas 5e.
# A L'ETAGE C : lam_true est la rente OPEN-LOOP. Les ecarts affiches ici sont donc
# ATTENDUS et valent le feedback lui-meme. Le juge est STEP 10, pas 5a/5e.
Sg = SA_T.clone().requires_grad_(True); tg = torch.arange(T)
_, Vg, _ = forward_all(Sg, tg)
gd = torch.diagonal(torch.stack([torch.autograd.grad(Vg[:, i].sum(), Sg, retain_graph=True)[0]
                                 for i in range(NAG)], dim=1), dim1=1, dim2=2).detach().numpy()
rel_g = np.abs(gd - lam_true)/np.maximum(lam_true, 1e-9)
print("\nSTEP 5e rente : dV/dS (tube) vs tete lambda, contre oracle")
for i in range(NAG):
    m = xa[:, i] > 1e-6
    if m.any():
        print(f"  a{i}: dV/dS med {np.median(rel_g[m,i]):.2%} max {rel_g[m,i].max():.2%}"
              f"  |  tete lam med {np.median(rel_lam[m,i]):.2%}  (SOB_OK={int(SOB_OK[i])})")
# profil temporel de la rente DEFLATEE : doit etre plate a la hauteur mu.
# NON INTERPRETABLE en zone kappa (FOC inactive, lambda non identifie).
lt = lam_net.numpy()/((1.0+r)**np.arange(T))[:, None]
for i in range(NAG):
    m = np.nonzero(xa[:, i] > 1e-6)[0]
    if len(m) > 6:
        q = [m[0], m[len(m)//4], m[len(m)//2], m[3*len(m)//4], m[-1]]
        print(f"  a{i} lam_deflate " + " ".join(f"t{t}:{lt[t,i]:.2f}" for t in q)
              + f"   (mu={mu_a[i]:.2f})")


# STEP 5b : champ stratifie
# field_int_med est la metrique de CHAMP la plus fiable, et le champ est l'argument meme
# du PINN face a une grille. La surveiller : une degradation du champ pendant que le
# sentier s'ameliore est la signature d'un surajustement au tube.
print(f"\nSTEP 5b champ (oracle tronque, n={CFG['n_field']})")
rng = np.random.default_rng(0)
Sf_ = rng.uniform(0.05, 1.10, (CFG["n_field"], NAG))*S0_np
tf_ = rng.integers(0, T-1, CFG["n_field"])
buck = {"kappa": [], "interieur": [], "zero": []}
t_f0 = time.time()
with torch.no_grad():
    for sv, tv in zip(Sf_, tf_):
        _, xo, _ = solve_oracle_N(sv, T-int(tv))
        xn, _, _ = forward_all(torch.tensor(sv).reshape(1, NAG), torch.tensor([int(tv)]), snap=True)
        for i in range(NAG):
            reg = ("kappa" if xo[0, i] >= KAP_np[i]-1e-6
                   else ("zero" if xo[0, i] <= 1e-9 else "interieur"))
            buck[reg].append(abs(xn[0, i].item()-xo[0, i]))
FIELD = {}
for reg, d in buck.items():
    if not d: continue
    d = np.array(d); FIELD[reg] = (float(np.median(d)), float(np.quantile(d, .9)))
    print(f"  {reg:9s} n={len(d):4d} | err med {FIELD[reg][0]:.3f} q90 {FIELD[reg][1]:.3f} bbl")
print(f"  ({time.time()-t_f0:.0f}s)")
FI = FIELD.get("interieur", (np.nan, np.nan))


# STEP 5c : recuperation hors-sentier
# distingue une POLITIQUE sigma(S) d'une sequence memorisee : partir d'ailleurs et
# retomber sur l'oracle recalcule depuis ce point.
print("\nSTEP 5c recuperation hors-sentier")
REC = {}
for fac in (0.8, 1.2):
    Xp, Pp, _ = rollout(S0V*fac)
    _, xo, po = solve_oracle_N(S0_np*fac, T, nbis=60, nouter=40)
    a_ = xo.sum(1) > 1e-6
    REC[fac] = float(np.abs(Pp[a_]-po[a_]).max()/po.max())
    print(f"  S0x{fac} : err_p={REC[fac]:.2e}")


# STEP 5d : decouplage 1-pas
# action du reseau A L'ETAT ORACLE de la date t, puis continuation ORACLE.
# Petit => la decision locale est bonne, l'erreur du rollout vient de la derive d'etat.
# Choisir des dates ou un agent est INTERIEUR : la ou tout le monde est colle a kappa, la
# decision est exacte par architecture et le test est degenere.
print("\nSTEP 5d decouplage 1-pas (etat oracle)")
DEC = {}
for ts in [int(u) for u in CFG["t_decouple"].split(",")]:
    if ts >= _tlast: continue
    Ss = SA_T[ts].reshape(1, NAG)
    with torch.no_grad():
        xs, _, _ = forward_all(Ss, torch.tensor([ts]), snap=True)
    xs = xs[0].numpy()
    _, _, ptail = solve_oracle_N(Ss[0].numpy()-xs, T-ts-1, nbis=60, nouter=40)
    ph = np.concatenate(([alpha - beta*xs.sum()], ptail))
    ao = xa[ts:].sum(1) > 1e-6
    DEC[ts] = float(np.abs(ph[ao]-pa[ts:][ao]).max()/pa.max())
    print(f"  t={ts:2d} : err_p={DEC[ts]:.2e}")


# STEP 6 : Bellman Monte-Carlo
# coherence de la VALEUR, SANS ORACLE : V_i(S0,0) predit contre le profit actualise
# REELLEMENT realise. Survit tel quel a l'etage C.
# QUASI-TAUTOLOGIQUE quand use_vmc=True : c'est exactement l'objet regresse sur le tube.
with torch.no_grad():
    _, V0, _ = forward_all(S0V.reshape(1, NAG), torch.tensor([0]))
d_t = delta**np.arange(T)
V_mc = np.array([float(np.sum(d_t*(P - C_np[i])*X[:, i])) for i in range(NAG)])
V_net = V0[0].numpy()
V_or = np.array([float(np.sum(d_t*(pa - C_np[i])*xa[:, i])) for i in range(NAG)])
print("  paiement realise vs oracle : " + " ".join(f"a{i} {(V_mc[i]/V_or[i]-1):+.3%}"
                                                   for i in range(NAG)))
BELL = float(np.max(np.abs(V_net-V_mc)/np.maximum(np.abs(V_mc), 1e-9)))
print(f"\nSTEP 6 Bellman MC : ecart max {BELL:.2%}")
for i in range(NAG):
    print(f"  a{i} : V_net={V_net[i]:9.1f} vs realise={V_mc[i]:9.1f}")


# STEP 7 : enveloppe et gradients croises
# COH est identiquement |R_lam|/lambda : il mesure l'erreur de la TETE LAMBDA, pas celle
# du gradient. Son estimand change avec SOB_OK : ne jamais comparer deux COH sous masques
# differents.
# AVERTISSEMENT : le SIGNE ne dit RIEN de la MAGNITUDE, et le feedback est un PRODUIT
# coef[i,k]*J[k,j]. 98% de signes corrects sont compatibles avec un facteur 3 d'erreur en
# amplitude. La magnitude est mesuree en 9f, pas ici.
Sp = (torch.rand(2000, NAG)*S0V*1.1).requires_grad_(True)
tp = torch.randint(0, T-1, (2000,))
_, Vp, lamp = forward_all(Sp, tp)
gVp = torch.stack([torch.autograd.grad(Vp[:, i].sum(), Sp, retain_graph=True)[0]
                   for i in range(NAG)], dim=1)

COH = (((torch.diagonal(gVp, dim1=1, dim2=2) - lamp).abs()
        / (lamp.abs() + 1e-6))[:, SOB_OK > 0]).median().item()

# filtre d'amplitude : on ignore les regions ou la derivee croisee est physiquement ~0.
# Sans ce filtre, CROSS_NEG comptait le signe d'une derivee nulle : quatre runs perdus.
off_mask = OFF.unsqueeze(0).expand_as(gVp) > 0
scale = lamp.detach().abs().unsqueeze(-1).expand_as(gVp)[off_mask] + 1e-9
offv = gVp[off_mask]
sig  = offv.abs() > 0.05 * scale
CROSS_NEG = (offv[sig] < 0).double().mean().item() if sig.any() else 0.0

print(f"\nSTEP 7 |dV_i/dS_i - lambda_i|/lambda = {COH:.2%}")
print(f"        dV_i/dS_j < 0 sur {CROSS_NEG:.0%} des cas significatifs (boite)")
for i in range(NAG):
    ci = ((gVp[:, i, i] - lamp[:, i]).abs()/(lamp[:, i].abs()+1e-6)).median().item()
    print(f"        COH a{i} = {ci:.2%}  (SOB_OK={int(SOB_OK[i])})")
for i in range(NAG):
    for j in range(NAG):
        if i == j: continue
        sig_ij = gVp[:, i, j].abs() > 0.05 * (lamp[:, i].detach().abs() + 1e-9)
        c_neg = (gVp[:, i, j][sig_ij] < 0).double().mean().item() if sig_ij.any() else 0.0
        print(f"        Boite (i={i}, j={j}) : {c_neg:.0%} negatifs (sur {int(sig_ij.sum())} pts)")

# mesure stricte sur le TUBE (seule zone d'interet pour le MPE)
St = SA_T.clone().requires_grad_(True)
tt = torch.arange(T)
_, Vt, _ = forward_all(St, tt)
gVt = torch.stack([torch.autograd.grad(Vt[:, i].sum(), St, retain_graph=True)[0]
                   for i in range(NAG)], dim=1)
for i in range(NAG):
    for j in range(NAG):
        if i == j: continue
        act_ij = (xa[:, i] > 1e-6) & (xa[:, j] > 1e-6)
        if act_ij.any():
            c_neg_t = (gVt[act_ij, i, j] < 0).double().mean().item()
            print(f"        Tube  (i={i}, j={j}) : {c_neg_t:.0%} negatifs (sur {int(act_ij.sum())} dates)")


# STEP 8 : taille du terme croise
# A cross=False c'est le terme OMIS (erreur de specification de l'open-loop) ; a cross=True
# c'est le terme effectivement internalise.
# ATTENTION : derivee LOCALE du premier ordre, identiquement nulle a un coin. Elle ne
# capture pas les deplacements de date de bascule de regime. Et le chiffre par date n'est
# PAS l'ecart d'equilibre : celui-ci est la somme actualisee (voir fb_ref / STEP 10).
FB = np.nan; FB_SIGN = np.nan
try:
    Sr = SA_T[:_tlast].clone().requires_grad_(True)
    tr = torch.arange(_tlast)
    xr, Vr, lamr = forward_all(Sr, tr)
    Sr2 = Sr - xr
    _, Vr2, _ = forward_all(Sr2, tr+1)
    Jr = torch.stack([torch.autograd.grad(xr[:, k].sum(), Sr, create_graph=True)[0]
                      for k in range(NAG)], dim=1)
    GV2r = torch.stack([torch.autograd.grad(Vr2[:, i].sum(), Sr2, create_graph=True)[0]
                        for i in range(NAG)], dim=1)
    pr = alpha - beta*xr.sum(-1, keepdim=True)
    coefr = -beta*xr.unsqueeze(-1) - delta*GV2r + torch.diag_embed(pr - COST)
    Mr = torch.einsum('bik,bkj->bij', coefr*OFF, Jr)   # feedback = somme k != i UNIQUEMENT
    fbs = torch.diagonal(Mr, dim1=1, dim2=2)*STRAT
    fb = fbs.abs().detach().numpy()
    rel = fb/np.maximum(np.abs(lamr.detach().numpy()), 1e-9)
    FB = float(np.median(rel[rel > 0])) if (rel > 0).any() else 0.0
    # le SIGNE decide si l'agent closed-loop est plus ou moins conservateur que l'open-loop.
    _fbs = fbs.detach().numpy()
    FB_SIGN = float(np.mean(_fbs[np.abs(_fbs) > 1e-9] < 0)) if (np.abs(_fbs) > 1e-9).any() else np.nan
    print(f"\nSTEP 8 |feedback_i| / lambda_i sur le sentier : mediane {FB:.2%}, max {rel.max():.2%}")
    print(f"        feedback < 0 sur {FB_SIGN:.0%} des points non nuls")
    for i in range(NAG):
        ri = rel[:, i]; ri = ri[ri > 0]
        if ri.size:
            print(f"  a{i} (STRAT={int(STRAT[i])}) : mediane {np.median(ri):.2%} max {ri.max():.2%}")
except Exception as e:
    print(f"\n[STEP 8 ignore : {e}]")

# diagnostic desagrege de lambda : interieur vs zone kappa. Cout nul.
# Un desaccord fort en zone kappa n'est PAS une regression : lambda n'y est pas identifie
# (FOC inactive). Les controles qui font foi : Hotelling interieur et dV/dS sur le tube.
for i in STRAT_IDX:
    m_int = INT[:, i]; m_kap = xa[:, i] >= KAP_np[i]-1e-3
    if m_int.any():
        print(f"  lam a{i} interieur : med {100*np.median(rel_lam[m_int,i]):.2f}%  "
              f"max {100*rel_lam[m_int,i].max():.2f}%  |  dV/dS interieur "
              f"{100*np.median(rel_g[m_int,i]):.2f}%", end="")
    if m_kap.any():
        print(f"  |  zone kappa {100*np.median(rel_lam[m_kap,i]):.2f}%")
    else:
        print()

# dates de BASCULE kappa -> interieur : LA ou une erreur de lambda issue de la chaine
# kappa entre dans une decision reellement LIBRE. Partout ailleurs le clamp l'absorbe.
# C'est le diagnostic qui a motive env_hard.
SWITCH = {}
for i in STRAT_IDX:
    sw = np.nonzero(INT[1:, i] & ~INT[:-1, i])[0] + 1
    if sw.size:
        SWITCH[i] = (float(rel_lam[sw, i].max()), float(rel_g[sw, i].max()))
        print(f"  a{i} bascule kappa->int t={list(sw)} : lam err max {100*SWITCH[i][0]:.2f}%"
              f"  |  dV/dS err max {100*SWITCH[i][1]:.2f}%")


# STEP 8b : ecran de feedback sur l'oracle seul (aucun reseau)
#   Repond a "l'etage C a-t-il un objet a mesurer ?" independamment de la qualite du PINN.
#   coef[i,k] = -beta*x_i - delta*dV_i/dS_k  et  J[k,i] = dx_k/dS_i, les deux par
#   differences CENTREES sur solve_oracle_N. feedback_i = somme_{k!=i} coef[i,k]*J[k,i].
#
#   LE CHIFFRE QUI COMPTE N'EST PAS fb/lambda A UNE DATE. La chaine d'enveloppe donne
#       lambda_i(0) = somme_t delta^t fb_i(t) + delta^T lambda_i(T)
#   donc l'ecart de RENTE entre closed-loop et open-loop est la SOMME ACTUALISEE des
#   feedbacks. Et comme lambda croit exactement au taux r, delta^t * fb_i(t) est CONSTANT
#   des lors que fb_i(t)/lambda_i(t) l'est : chaque date interieure contribue le MEME
#   montant, sans attenuation. Un feedback de 0.3% par date sur 20 dates de recouvrement
#   donne ~6% sur mu, pas 0.3%. C'est ce ratio CUMULE qu'il faut comparer au plancher de
#   bruit du solveur (err_p_noext).
#
#   BALAYER TOUTES LES DATES DE RECOUVREMENT : fb est identiquement nul des qu'un
#   strategique est a un coin. Trois dates tirees au hasard renvoient surtout des zeros
#   STRUCTURELS et sous-estiment massivement (mesure : t=0 et t=20 -> 0.00%, t=10 -> 0.29%).
#
#   CONSEQUENCE SUR T : reduire l'horizon reduit le nombre de dates de recouvrement, donc
#   reduit le signal PROPORTIONNELLEMENT. Ne pas raccourcir T pour gagner du temps de
#   calcul : c'est la seule quantite que le projet cherche a mesurer. Le biais de lambda,
#   lui, se corrige par env_hard, pas par T.
def feedback_screen(dates=None, h_rel=None, verbose=True):
    h_rel = CFG["fb_h"] if h_rel is None else h_rel
    if dates is None:
        need = 2 if len(STRAT_IDX) >= 2 else 1
        dates = list(np.nonzero(INT[:, STRAT_IDX].sum(1) >= need)[0])
    dt = delta**np.arange(T)
    cum = np.zeros(NAG); rows = []
    for t0 in dates:
        S0t = SA_T[t0].numpy(); Th = T - int(t0)
        if Th < 3: continue
        h = np.minimum(np.maximum(h_rel*S0_np, 1e-6), 0.4*np.maximum(S0t, 1e-9))
        _, x0, p0 = solve_oracle_N(S0t, Th, nbis=60, nouter=40)
        Jc = np.zeros((NAG, NAG)); dV = np.zeros((NAG, NAG))
        for k in range(NAG):
            if h[k] <= 1e-9: continue
            Sp = S0t.copy(); Sp[k] += h[k]
            Sm = S0t.copy(); Sm[k] -= h[k]          # borne symetrique : 2h reste exact
            _, xp, pp = solve_oracle_N(Sp, Th, nbis=60, nouter=40)
            _, xm, pm = solve_oracle_N(Sm, Th, nbis=60, nouter=40)
            Jc[:, k] = (xp[0] - xm[0])/(2*h[k])
            for i in range(NAG):
                dV[i, k] = (np.sum(dt[:Th]*(pp - C_np[i])*xp[:, i])
                            - np.sum(dt[:Th]*(pm - C_np[i])*xm[:, i]))/(2*h[k])
        coef = -beta*x0[0][:, None] - delta*dV
        fb = ((coef*(1.0-np.eye(NAG))) @ Jc).diagonal().copy()*STRAT.numpy()
        cum += dt[int(t0)]*fb                       # somme actualisee -> ecart sur mu
        rows.append((int(t0), fb/np.maximum(mu_a*(1.0+r)**int(t0), 1e-9)))
    rel_cum = cum/np.maximum(mu_a, 1e-9)
    if verbose:
        print("\nETAPE 8b  ecran de feedback (ORACLE seul, aucun reseau)")
        print(f"  {len(rows)} dates de recouvrement balayees, h_rel={h_rel:.1e}")
        for t0, rl in rows[::max(1, len(rows)//6)]:
            print(f"    t={t0:2d} fb/lambda = " + "  ".join(f"a{i} {rl[i]:+.3%}" for i in STRAT_IDX))
        print("  ECART CUMULE SUR LA RENTE (somme_t delta^t fb / mu) :")
        for i in STRAT_IDX:
            print(f"    a{i} : {rel_cum[i]:+.2%}")
        mx = max(abs(rel_cum[i]) for i in STRAT_IDX) if STRAT_IDX else 0.0
        print(f"  max |ecart| = {mx:.2%}   vs plancher de bruit du solveur {ERRP_NOEXT:.2%}")
        print("    " + ("SIGNAL EXPLOITABLE : l'etage C est falsifiable."
                        if mx > 3*ERRP_NOEXT else
                        "SIGNAL TROP FAIBLE : l'etage C sera infalsifiable. Recalibrer pour "
                        "AUGMENTER le recouvrement interieur (surtout PAS reduire T)."))
    return rel_cum, rows

FBSCR = FB_REF.copy()          # prediction figee ; recalculee si run_fb_screen=True
if CFG["run_fb_screen"]:
    try:
        FBSCR, _fbrows = feedback_screen()
    except Exception as e:
        print(f"\n[ETAPE 8b ignoree : {e}]")

# 8b-h : BALAYAGE EN h. Test de verite de fb_ref, qui n'en avait jamais passe. La cible
# de fb_ref etant une difference finie centree sur solve_oracle_N, un chiffre qui bouge
# avec le pas ne mesure pas le feedback mais l'erreur de troncature. Aucun reseau.
FBH_SPREAD = np.nan
if CFG["fb_h_sweep"]:
    try:
        print("\n 8b-h  sensibilite de fb_ref au pas de difference finie (ORACLE seul)")
        _hs = [float(u) for u in CFG["fb_h_list"].split(",")]
        _sw = {}
        for _h in _hs:
            _sw[_h] = feedback_screen(h_rel=_h, verbose=False)[0]
            print(f"    h={_h:.1e} : "
                  + "  ".join(f"a{i} {_sw[_h][i]:+.2%}" for i in STRAT_IDX)
                  + ("   <- fb_h courant" if abs(_h-CFG["fb_h"]) < 1e-12 else ""))
        _ref = _sw.get(CFG["fb_h"], _sw[_hs[len(_hs)//2]])
        if STRAT_IDX:
            # NORMALISER PAR AGENT. Diviser par le max global fait juger le petit agent a
            # l'echelle du grand : a0 sortait a 3% alors qu'il est a 15% de son propre
            # niveau. C'est le meme travers d'agregation que 9h avant sa correction.
            _sp = {i: max(abs(v[i]-_ref[i]) for v in _sw.values())/max(abs(_ref[i]), 1e-12)
                   for i in STRAT_IDX}
            FBH_SPREAD = max(_sp.values())
            FBSCR = _ref.copy()
            print("    sensibilite relative PAR AGENT : "
                  + "  ".join(f"a{i} {_sp[i]:.0%}" for i in STRAT_IDX))
            print("    " + ("REFERENCE VALIDE sur tous les strategiques."
                            if FBH_SPREAD < 0.10 else
                            f"REFERENCE FRAGILE : au moins un agent a {FBH_SPREAD:.0%} de "
                            "sensibilite au pas ; le lire comme un ordre de grandeur."))
            print(f"    rappel fb_ref fige dans CFG : "
                  + "  ".join(f"a{i} {FB_REF[i]:+.2%}" for i in STRAT_IDX))
    except Exception as e:
        print(f"\n[8b-h ignore : {e}]")


# STEP 9 : batterie de coherence SANS ORACLE (en-tete)
#
#   C'est le validateur destine a survivre a la disparition de l'oracle (etage C, puis
#   N > 3 ou aucune grille n'existe). Chaque test ci-dessous se suffit a lui-meme : il ne
#   compare le reseau qu'a des quantites qu'il produit lui-meme, ou a des identites que
#   tout equilibre doit satisfaire.
#
#   REGLE DE LECTURE : un reseau irreprochable sur 9a-9g n'est pas garanti correct, mais
#   un reseau qui echoue a l'un d'eux est certainement faux. C'est une batterie de
#   FALSIFICATION, pas une preuve.
STEP9 = {}
if CFG["run_step9"]:
    print("\n" + "="*70)
    print("STEP 9  COHERENCE SANS ORACLE")
    print("="*70)


# 9a : faisabilite
if CFG["run_step9"]:
    # x >= 0, x <= kappa, x <= S, et stock final >= 0. Garanti par l'architecture
    # (min(min(S,kappa), softplus)), donc un echec ici est un bug de cablage, pas
    # d'apprentissage. Test a cout nul qu'il faut garder pour cette raison meme.
    feas = dict(neg=float(X.min()), over_kap=float((X - KAP_np[None, :]).max()),
                send=float(Send.min()), unext=float((S0_np - X.sum(0)).max()))
    print(f"\n 9a faisabilite : min(x)={feas['neg']:.2e}  max(x-kappa)={feas['over_kap']:.2e}"
          f"  min(S_T)={feas['send']:.2e}  stock non extrait max={feas['unext']:.3f}")
    print(f"    {'OK' if feas['neg']>-1e-12 and feas['over_kap']<1e-9 and feas['send']>-1e-9 else 'ECHEC'}"
          f"  (le stock non extrait est une PERTE : valeur terminale nulle)")
    STEP9["feas_unext"] = feas["unext"]


# 9b : Euler / Hotelling sans mu
if CFG["run_step9"]:
    # A l'interieur, h_i = p - beta*Lambda_i - c_i doit croitre EXACTEMENT au taux r.
    # On teste le RATIO h_i(t+1)/h_i(t) contre (1+r) sur les dates interieures
    # CONSECUTIVES. Aucun mu, aucun oracle : c'est la condition d'arbitrage elle-meme.
    # A l'etage C ce ratio s'ecarte de (1+r), et l'ecart EST le feedback.
    print("\n 9b Euler/Hotelling sans oracle : h_i(t+1)/h_i(t) doit valoir 1+r =", f"{1+r:.3f}")
    EUL = {}
    hall_raw = P[:, None] - beta*(X @ A_np.T) - C_np[None, :]
    for i in range(NAG):
        mi = (X[:, i] > 1e-3) & (X[:, i] < KAP_np[i]-1e-3)
        idx = np.nonzero(mi)[0]
        pair = [k for k in idx if (k+1) in idx and hall_raw[k, i] > 1e-6]
        if len(pair) < 2: continue
        rat = hall_raw[np.array(pair)+1, i]/hall_raw[np.array(pair), i]
        EUL[i] = (float(np.median(np.abs(rat/(1+r)-1))), float(np.abs(rat/(1+r)-1).max()))
        print(f"    a{i} : ecart median {EUL[i][0]:.2%}  max {EUL[i][1]:.2%}  "
              f"sur {len(pair)} paires interieures consecutives")
    STEP9["euler_med"] = float(np.median([v[0] for v in EUL.values()])) if EUL else np.nan


# 9c : Bellman par agent et hors-sentier
if CFG["run_step9"]:
    # V_i(S,t) contre le profit actualise realise en deroulant la politique depuis (S,t).
    # Evalue sur des etats TIRES AU HASARD, pas seulement sur S0 : c'est ce qui le rend
    # non tautologique meme avec use_vmc=True (la cible MC ne vit que sur le tube).
    _rng9 = np.random.default_rng(1)
    Sb9 = torch.tensor(_rng9.uniform(0.15, 1.10, (128, NAG))*S0_np)
    tb9 = torch.tensor(_rng9.integers(0, T-2, 128))
    with torch.no_grad():
        _, Vnet9, _ = forward_all(Sb9, tb9)
    Vmc9 = mc_value(Sb9, tb9)
    relb = (Vnet9 - Vmc9).abs()/Vmc9.abs().clamp(min=1e-6)
    # STRATIFIER PAR t : aux dates tardives V -> 0 et l'erreur RELATIVE explose
    # mecaniquement. Un q90 global y est ininterpretable ; c'est le tiers precoce qui
    # porte l'information (V_i(S,0) est l'objet economiquement significatif).
    print("\n 9c Bellman hors-sentier (V_net vs profit realise, 128 etats aleatoires)")
    _b3 = [(tb9 < T//3), (tb9 >= T//3) & (tb9 < 2*T//3), (tb9 >= 2*T//3)]
    _nm = ["t<T/3 ", "T/3-2T/3", "t>2T/3"]
    for i in range(NAG):
        line = f"    a{i} : "
        for m_, nm_ in zip(_b3, _nm):
            if m_.sum() < 3: continue
            line += f"{nm_} med {relb[m_,i].median():6.2%} q90 {relb[m_,i].quantile(.9):7.2%}  |  "
        print(line)
    STEP9["bell_off_med"] = float(relb[_b3[0]].median()) if _b3[0].sum() else float(relb.median())


# 9d : deviation a un coup
if CFG["run_step9"]:
    #   DEFINITION de l'equilibre : aucun joueur ne gagne a devier une fois puis a
    #   reprendre sa politique. On perturbe x_i(t0) de eps, on continue avec les
    #   politiques apprises, et on compare le profit actualise de i a celui de eps = 0.
    #   Le gain maximal doit etre ~0 (a l'ordre de la resolution en eps).
    #
    #   freeze=True  : les RIVAUX gardent leurs quantites de la trajectoire de reference
    #                  -> condition d'equilibre OPEN-LOOP. Doit etre nulle aux etages A/B.
    #   freeze=False : les rivaux REPONDENT via leur politique sigma_k(S)
    #                  -> condition d'equilibre MARKOVIEN. Doit etre nulle a l'etage C.
    #   L'ECART ENTRE LES DEUX GAINS EST LA DIVERGENCE CL/OL, mesuree sans aucune grille
    #   ni oracle. C'est le second estimateur, independant de la grille.
    #
    #   Un gain positif significatif en freeze=True a l'etage A signale soit une politique
    #   non optimale, soit une valeur mal ancree. Un gain de l'ordre de 1e-4 est le bruit
    #   de discretisation en eps.
    def deviation_gain(t_probe, freeze=True, eps_rel=None, n_eps=None):
        eps_rel = CFG["dev_eps_rel"] if eps_rel is None else eps_rel
        n_eps   = CFG["dev_n_eps"]   if n_eps   is None else n_eps
        with torch.no_grad():
            S = S0V.clone().reshape(1, NAG); XB = []
            for t in range(T):
                xb, _, _ = forward_all(S, torch.tensor([t]), snap=True)
                XB.append(xb[0].clone()); S = S - xb
            XB = torch.stack(XB)                                   # [T,NAG] reference
            SB = S0V.unsqueeze(0) - torch.cat([torch.zeros(1, NAG),
                                               torch.cumsum(XB, 0)[:-1]], 0)
            out = np.full((len(t_probe), NAG), np.nan)
            M = n_eps; mid = n_eps//2
            for a_, t0 in enumerate(t_probe):
                for i in range(NAG):
                    capi = float(torch.minimum(SB[t0, i], KAP[i]))
                    if capi <= 1e-6: continue
                    e = torch.linspace(-eps_rel*capi, eps_rel*capi, M)
                    S = SB[t0].expand(M, NAG).clone()
                    prof = torch.zeros(M)
                    for t in range(t0, T):
                        ti = torch.full((M,), t, dtype=torch.long)
                        xp, _, _ = forward_all(S, ti, snap=True)
                        x = xp.clone()
                        if freeze:
                            for k in range(NAG):
                                if k != i: x[:, k] = XB[t, k]
                        if t == t0:
                            x[:, i] = torch.clamp(XB[t0, i] + e, min=0.0)
                        x = torch.minimum(torch.minimum(x, S), KAP)
                        p = alpha - beta*x.sum(-1)
                        prof = prof + (delta**(t-t0))*(p - COST[i])*x[:, i]
                        S = S - x
                    base = float(prof[mid])
                    out[a_, i] = float((prof.max() - prof[mid]))/max(abs(base), 1e-9)
            return out, XB

    # dates sondees : celles ou le maximum d'agents STRATEGIQUES sont interieurs.
    _score = INT[:, STRAT_IDX].sum(1) if STRAT_IDX else INT.sum(1)
    _cands = np.nonzero(_score >= max(1, _score.max()))[0]
    if _cands.size >= CFG["dev_n_dates"]:
        t_probe = list(np.linspace(_cands[0], _cands[-1], CFG["dev_n_dates"]).astype(int))
    else:
        t_probe = list(np.linspace(0, max(1, _tlast-1), CFG["dev_n_dates"]).astype(int))
    if CFG["dev_skip_ext"]:
        # sans ce filtre, le max est pilote par la date d'extinction d'un agent, exactement
        # le defaut deja corrige sur err_p. A l'etage C cela rendrait 9d illisible.
        _tp = [u for u in t_probe if not _ext[u]]
        if len(_tp) >= 2: t_probe = _tp
    t_probe = sorted(set(int(u) for u in t_probe))
    print(f"\n 9d test de deviation a un coup (dates {t_probe}, "
          f"eps = +/-{CFG['dev_eps_rel']:.0%} de la capacite du pas)")
    DEV_F, _XB = deviation_gain(t_probe, freeze=True)
    DEV_M, _   = deviation_gain(t_probe, freeze=False)
    print("      gain relatif max ; FREEZE = condition open-loop, MARKOV = condition MPE")
    print("    t   | " + " | ".join(f"a{i} freeze / markov" for i in range(NAG)))
    for a_, t0 in enumerate(t_probe):
        s = " | ".join(f"{DEV_F[a_,i]:9.2e} /{DEV_M[a_,i]:9.2e}" for i in range(NAG))
        print(f"   {t0:3d} | {s}")
    DEVF = float(np.nanmax(DEV_F[:, STRAT_IDX])) if STRAT_IDX else float(np.nanmax(DEV_F))
    DEVM = float(np.nanmax(DEV_M[:, STRAT_IDX])) if STRAT_IDX else float(np.nanmax(DEV_M))
    print(f"    max sur les strategiques : freeze {DEVF:.2e}  |  markov {DEVM:.2e}")
    if not CFG["cross"]:
        print(f"    ETAGE {'B' if CFG['sob_off'] else 'A'} : c'est FREEZE qui doit etre nul.")
        print(f"    DIVERGENCE CL/OL implicite = {abs(DEVM-DEVF):.2e} "
              f"(gain markovien laisse sur la table par la solution open-loop)")
    else:
        # Le critere sur le SIGNE seul est un faux positif garanti quand les deux gains
        # sont au plancher de bruit : a l'etage C, 4.12e-5 < 4.33e-5 a ete imprime
        # "ORDRE CORRECT" alors que 5% d'ecart entre deux quantites de 4e-5 est du bruit
        # de discretisation en eps. Le critere doit exiger un FACTEUR.
        RATIO = DEVF/max(DEVM, 1e-12)
        print(f"    ETAGE C : c'est MARKOV qui doit etre nul.")
        print(f"    DIVERGENCE CL/OL implicite = {abs(DEVF-DEVM):.2e}")
        print(f"    ratio freeze/markov = {RATIO:.2f}  -> "
              + ("BASCULE CONFIRMEE" if RATIO > 3.0 else
                 "NON CONCLUANT : les deux gains sont au plancher de bruit"))
        STEP9["dev_ratio"] = RATIO
    STEP9["dev_freeze"] = DEVF; STEP9["dev_markov"] = DEVM


# 9e : symetrie
if CFG["run_step9"]:
    # Si deux agents ont des parametres identiques, leurs politiques DOIVENT coincider.
    # Test sans oracle, tres discriminant : il attrape tout bug de cablage asymetrique.
    # Inactif si aucune paire symetrique dans la calibration.
    SYM = np.nan
    _pairs = [(i, j) for i in range(NAG) for j in range(i+1, NAG)
              if abs(C_np[i]-C_np[j]) < 1e-9 and abs(KAP_np[i]-KAP_np[j]) < 1e-9
              and abs(S0_np[i]-S0_np[j]) < 1e-9 and BLOC[i] != BLOC[j]]
    if _pairs:
        Ss = torch.tensor(_rng9.uniform(0.15, 1.05, (256, NAG))*S0_np)
        ts = torch.tensor(_rng9.integers(0, T-1, 256))
        with torch.no_grad():
            xs, _, _ = forward_all(Ss, ts, snap=True)
            d = []
            for (i, j) in _pairs:
                Sw = Ss.clone(); Sw[:, [i, j]] = Ss[:, [j, i]]
                xw, _, _ = forward_all(Sw, ts, snap=True)
                d.append(((xs[:, i]-xw[:, j]).abs()/XREF[i]).median().item())
        SYM = float(np.max(d))
        print(f"\n 9e symetrie (paires {_pairs}) : ecart median max {SYM:.2%} de la capacite")
    else:
        print("\n 9e symetrie : aucune paire d'agents symetriques dans cette calibration (test inactif)")
    STEP9["sym"] = SYM


# 9f : magnitude de gV vs cible pathwise
if CFG["run_step9"]:
    #   gV_target[i,j] = (V_i^MC(S + D e_j, t) - V_i^MC(S - D e_j, t)) / (2D), la politique
    #   COURANTE etant deroulee jusqu'a T. C'est exactement la derivee totale markovienne,
    #   reponses de politique comprises. Cible MESUREE, non auto-referentielle : c'est
    #   l'attaque de P-C (la cible de R_soff est construite depuis le V du reseau lui-meme).
    #   PREMIERE mesure de MAGNITUDE de gV du projet. Le signe (98-100% negatifs) n'en
    #   disait rien, et le feedback est un PRODUIT.
    #   CRITERE : < 10% sur les couples (i,j) strategiques. Au-dela, l'etage C est premature.
    #   freeze=True gele les rivaux -> gradient OPEN-LOOP. La difference des deux versions
    #   est un TROISIEME estimateur de la divergence CL/OL.

    n9 = min(_tlast, 30)
    S9 = SA_T[:n9].clone().requires_grad_(True); t9 = torch.arange(n9)
    _, V9, _ = forward_all(S9, t9)
    gV9 = torch.stack([torch.autograd.grad(V9[:, i].sum(), S9, retain_graph=True)[0]
                       for i in range(NAG)], dim=1).detach()
    GT_cl = gv_target_fd(SA_T[:n9], torch.arange(n9), freeze=False)
    print("\n 9f magnitude de gV contre cible pathwise mesuree (tube, differences centrees)")
    GVERR = {}
    for i in range(NAG):
        for j in range(NAG):
            m = (xa[:n9, i] > 1e-6) & (xa[:n9, j] > 1e-6)
            if m.sum() < 3: continue
            num = (gV9[m, i, j] - GT_cl[m, i, j]).abs()
            den = GT_cl[m, i, j].abs().clamp(min=1e-6)
            v = float((num/den).median())
            GVERR[(i, j)] = v
            tag = "diag" if i == j else ("CROISE STRAT" if (i in STRAT_IDX and j != i) else "croise")
            print(f"    gV({i},{j}) [{tag:12s}] : err med {v:6.1%}   "
                  f"| reseau {float(gV9[m,i,j].median()):9.3f}  cible {float(GT_cl[m,i,j].median()):9.3f}")
    _crit = [v for (i, j), v in GVERR.items() if i != j and i in STRAT_IDX]
    STEP9["gv_cross_err"] = float(np.max(_crit)) if _crit else np.nan
    if _crit:
        print(f"    CRITERE : max sur les croises strategiques = {max(_crit):.1%} "
              f"{'OK (< 10%)' if max(_crit) < 0.10 else '-> ETAGE C PREMATURE, basculer R_soff sur cette cible'}")


# 9g : magnitude de J vs differences finies
if CFG["run_step9"]:
    # J = dx_k/dS_j par autodiff, contre (x_k(S+D e_j) - x_k(S-D e_j))/(2D) du reseau
    # lui-meme. Ne teste pas l'economie mais le CABLAGE de l'autodiff, qui n'a jamais ete
    # verifie alors que J entre directement dans le feedback.
    if CFG["s9_J"]:
      with torch.no_grad():
        dl = CFG["fd_rel"]*S0V
        Jfd = torch.zeros(n9, NAG, NAG)
        for j in range(NAG):
            ej = torch.zeros(NAG); ej[j] = 1.0
            xp_, _, _ = forward_all(SA_T[:n9] + ej*dl, t9)
            xm_, _, _ = forward_all(torch.clamp(SA_T[:n9] - ej*dl, min=0.0), t9)
            Jfd[:, :, j] = (xp_ - xm_)/(2*dl[j])
      S9b = SA_T[:n9].clone().requires_grad_(True)
      x9, _, _ = forward_all(S9b, t9)
      Jad = torch.stack([torch.autograd.grad(x9[:, k].sum(), S9b, retain_graph=True)[0]
                         for k in range(NAG)], dim=1).detach()
      mfree = torch.tensor(INT[:n9].astype(float))        # J n'est non nul qu'a l'interieur
      num = (Jad - Jfd).abs(); den = Jfd.abs().clamp(min=1e-4)
      print("\n 9g coherence de J (autodiff vs differences finies sur la politique)")
      for k in range(NAG):
          m = mfree[:, k] > 0
          if m.sum() < 3: continue
          print(f"    J({k},:) sur {int(m.sum())} dates interieures : err med "
                f"{float((num[m,k,:]/den[m,k,:]).median()):.1%}  "
                f"|J| med {float(Jad[m,k,:].abs().median()):.4f}")
      STEP9["J_err"] = float((num[mfree.sum(1) > 0]/den[mfree.sum(1) > 0]).median())
    else:
        print("\n 9g coherence de J (autodiff vs FD du reseau) : DESACTIVE, valide 4.7e-5")


# 9h : J du reseau vs J de l'oracle
if CFG["run_step9"]:
    # 9g ne compare le reseau qu'a LUI-MEME : il valide le CABLAGE de l'autodiff, pas
    # l'ECONOMIE. Ici on compare dx_k/dS_j du reseau a la meme derivee calculee par
    # differences CENTREES sur solve_oracle_N. C'est la seule verification que J porte
    # bien la reponse strategique reelle, et J entre directement dans le feedback.
    # A l'etage C l'oracle n'est plus l'equilibre : l'ecart attendu est alors de l'ordre
    # du feedback lui-meme, pas zero. Lire ce test comme un ORDRE DE GRANDEUR.
    if CFG["s9_Jor"]:
        _dj = [int(u) for u in np.linspace(0, max(1, len(np.nonzero(
            INT[:, STRAT_IDX].sum(1) >= max(1, len(STRAT_IDX)))[0])-1),
            CFG["s9_Jor_dates"]).astype(int)]
        _cand = np.nonzero(INT[:, STRAT_IDX].sum(1) >= max(1, len(STRAT_IDX)))[0]
        _dj = sorted(set(int(_cand[u]) for u in _dj if u < len(_cand))) if len(_cand) else []
        print("\n 9h J du reseau vs J de l'ORACLE, PAIRES CONJOINTEMENT INTERIEURES")
        print("     le HORS-DIAGONAL J[k,i] est la seule entree qui alimente le feedback ;")
        print("     l'ancienne agregation (max sans masque de regime) a produit le faux")
        print("     diagnostic d'une Jacobienne effondree a 85-100%.")
        JOFF, JSGN, JDIA, JANA, JANS = [], [], [], [], []
        for t0 in _dj:
            S0t = SA_T[t0].numpy(); Th = T - t0
            if Th < 3: continue
            hh = np.minimum(np.maximum(CFG["fb_h"]*S0_np, 1e-6), 0.4*np.maximum(S0t, 1e-9))
            Jo = np.zeros((NAG, NAG))
            for k in range(NAG):
                # un agent epuise a S_k ~ 0 voit son pas ecrase a 4e-10 par la borne
                # 0.4*max(S0t, 1e-9) : la difference divisee est alors du bruit pur.
                # C'est la source du faux diagnostic "Jacobienne effondree a 85-100%".
                if hh[k] <= 1e-9: continue
                Sp_ = S0t.copy(); Sp_[k] += hh[k]
                Sm_ = S0t.copy(); Sm_[k] -= hh[k]
                _, xpo, _ = solve_oracle_N(Sp_, Th, nbis=60, nouter=40)
                _, xmo, _ = solve_oracle_N(Sm_, Th, nbis=60, nouter=40)
                Jo[:, k] = (xpo[0] - xmo[0])/(2*hh[k])
            Sn_ = SA_T[t0:t0+1].clone().requires_grad_(True)
            xn_, _, _ = forward_all(Sn_, torch.tensor([t0]))
            Jn = torch.stack([torch.autograd.grad(xn_[:, k].sum(), Sn_, retain_graph=True)[0]
                              for k in range(NAG)], dim=1).detach().numpy()[0]
            # 9h-bis : J ANALYTIQUE. Au lieu de LIRE la jacobienne dans la tete
            # politique, on la RESOUT. A l'equilibre interieur h_i == 0 sur un ouvert,
            # donc dh_i/dS_j = 0, soit (delta*L - beta*B) J = delta*L, avec
            # L[i,k] = dlambda'_i/dS'_k et B = 1 + A. Les lignes des agents CONTRAINTS
            # sont connues (0 a kappa ou a zero, e_k en regime x=S) : on les passe au
            # second membre et on ne resout QUE le bloc des agents interieurs. Masquer
            # les colonnes sans masquer les lignes etait le bug qui avait fait rejeter
            # cette piste. Interet : L derive d'une tete LISSE ajustee a ~1.5 %, contre
            # une tete politique a kinks ajustee a ~6 % -- signal/bruit ~8x meilleur.
            Ja = None
            if CFG["s9_Jana"]:
                Sa_ = SA_T[t0:t0+1].clone().requires_grad_(True)
                xa_, _, _ = forward_all(Sa_, torch.tensor([t0]))
                S2a_ = Sa_ - xa_
                _, _, lam2a_ = forward_all(S2a_, torch.tensor([min(t0+1, T)]))
                Lm = torch.stack([torch.autograd.grad(lam2a_[:, i].sum(), S2a_,
                                                      retain_graph=True)[0]
                                  for i in range(NAG)], dim=1).detach().numpy()[0]
                xn0 = xa_.detach().numpy()[0]; Sn0 = SA_T[t0].numpy()
                at_k = xn0 >= KAP_np - 1e-4
                at_s = (~at_k) & (xn0 >= Sn0 - 1e-4)
                fr_ = ~(at_k | at_s | (xn0 <= 1e-6))
                Jk = np.zeros((NAG, NAG))
                for k in np.nonzero(at_s)[0]: Jk[k, k] = 1.0
                Mm = delta*Lm - beta*(1.0 + A_np)
                Ja = Jk.copy()
                if fr_.any():
                    Fi = np.nonzero(fr_)[0]; Fc = np.nonzero(~fr_)[0]
                    rhs = delta*Lm[Fi, :]
                    if Fc.size: rhs = rhs - Mm[np.ix_(Fi, Fc)] @ Jk[Fc, :]
                    try:
                        Ja[Fi, :] = np.linalg.solve(Mm[np.ix_(Fi, Fi)], rhs)
                    except np.linalg.LinAlgError:
                        Ja = None

            # RESTRICTION AUX PAIRES CONJOINTEMENT INTERIEURES. L'ancienne version
            # prenait un max sur toutes les paires sans masque de regime : elle melangeait
            # un agent epuise et un coin x=S a une Jacobienne parfaitement correcte.
            int_t = INT[t0]                       # regimes de l'ORACLE a cette date
            # REGIME x = S : l'oracle donne dx_k/dS_k = 1 exactement quand l'agent epuise
            # son stock residuel. INT ne le detecte pas (il ne teste que 0 < x < kappa).
            # Le reseau y place la bascule une periode plus tard : c'est le biais de date
            # d'extinction, connu depuis M2, PAS un defaut de Jacobienne. On l'exclut --
            # sans ce masque, J_diag_err sortait a 84.8% sur le seul point t=35.
            xS = np.diag(Jo) > 0.9
            print(f"    t={t0:2d}  interieurs oracle : "
                  + " ".join(f"a{i}" for i in range(NAG) if int_t[i])
                  + ("   [x=S exclu : "
                     + " ".join(f"a{i}" for i in range(NAG) if xS[i]) + "]" if xS.any() else ""))
            for i in STRAT_IDX:
                if not int_t[i] or xS[i]: continue
                if abs(Jo[i, i]) > 1e-4:
                    ed = abs(Jn[i, i]-Jo[i, i])/abs(Jo[i, i]); JDIA.append(ed)
                    print(f"      J[{i},{i}] diag      : reseau {Jn[i,i]:+.5f}  "
                          f"oracle {Jo[i,i]:+.5f}  ecart {ed:.1%}")
                for k in range(NAG):
                    # seule entree qui alimente le feedback : k != i, les deux interieurs,
                    # et une derivee oracle non nulle (sinon on mesure le signe de zero).
                    if k == i or not int_t[k] or xS[k] or abs(Jo[k, i]) < 1e-4: continue
                    eo = abs(Jn[k, i]-Jo[k, i])/abs(Jo[k, i])
                    sg = bool(Jn[k, i]*Jo[k, i] > 0)
                    JOFF.append(eo); JSGN.append(sg)
                    print(f"      J[{k},{i}] HORS-DIAG : reseau {Jn[k,i]:+.5f}  "
                          f"oracle {Jo[k,i]:+.5f}  ecart {eo:.1%}  signe "
                          + ("OK" if sg else "FAUX"))
                    if Ja is not None:
                        ea = abs(Ja[k, i]-Jo[k, i])/abs(Jo[k, i])
                        sa = bool(Ja[k, i]*Jo[k, i] > 0)
                        JANA.append(ea); JANS.append(sa)
                        print(f"                 ANALYTIQUE {Ja[k,i]:+.5f}"
                              f"                    ecart {ea:.1%}  signe "
                              + ("OK" if sa else "FAUX"))
        # MEDIANE et non max : un max sur un agregat est pilote par son point le plus
        # degenere. C'est le defaut de 9h que l'on vient de corriger un cran plus haut.
        STEP9["J_diag_med"] = float(np.median(JDIA)) if JDIA else np.nan
        STEP9["J_diag_max"] = float(np.max(JDIA)) if JDIA else np.nan
        if JDIA:
            print(f"    DIAGONALE sur {len(JDIA)} entrees : ecart median "
                  f"{np.median(JDIA):.1%}, max {np.max(JDIA):.1%}")
        if JOFF:
            _med = float(np.median(JOFF)); _bad = 1.0 - float(np.mean(JSGN))
            STEP9["J_offdiag_med"] = _med
            STEP9["J_offdiag_max"] = float(np.max(JOFF))
            STEP9["J_offdiag_signbad"] = _bad
            print(f"    HORS-DIAGONAL sur {len(JOFF)} paires : ecart median {_med:.1%}, "
                  f"max {np.max(JOFF):.1%}, signe faux sur {_bad:.0%}")
            print("    " + ("J HORS-DIAG DISCULPE : le diagnostic est a refaire, le suspect "
                            "devient l'injection dans la loss."
                            if _med < 0.20 and _bad < 0.10 else
                            "J HORS-DIAG INCRIMINE : passer au residu R_dfoc (action 3), "
                            "valide a l'etage A d'abord."))
        else:
            print("    aucune paire conjointement interieure exploitable a ces dates : "
                  "augmenter s9_Jor_dates.")
        if JANA:
            _ma = float(np.median(JANA)); _ba = 1.0 - float(np.mean(JANS))
            STEP9["Jana_offdiag_med"] = _ma
            STEP9["Jana_offdiag_signbad"] = _ba
            print(f"    ANALYTIQUE sur {len(JANA)} paires : ecart median {_ma:.1%}, "
                  f"signe faux sur {_ba:.0%}"
                  + (f"   (autodiff : {np.median(JOFF):.1%} / {1.0-np.mean(JSGN):.0%})"
                     if JOFF else ""))
            if JOFF:
                print("    " + ("PISTE OUVERTE : resoudre J bat le lire dans le reseau."
                                if _ma < 0.5*np.median(JOFF) else
                                "PISTE FERMEE : resoudre J n'apporte rien, L est aussi "
                                "mauvais que la tete politique."))


# 9 : synthese
if CFG["run_step9"]:
    print("\n  SYNTHESE STEP 9 (tout sans oracle) : "
          + " | ".join(f"{k}={v:.2e}" for k, v in STEP9.items() if v == v))
    print("="*70)


# STEP 10 : juge quantitatif de l'etage C
#   L'oracle open-loop n'est plus validateur, mais on dispose de TROIS predictions
#   preenregistrees, toutes independantes du reseau :
#     (1) le gain de deviation MARKOVIEN doit tomber a ~0, et le gain FREEZE doit MONTER
#         (l'open-loop cesse d'etre un equilibre quand les rivaux repondent) ;
#     (2) la rente initiale doit se deplacer de fb_ref, mesure sur l'oracle seul ;
#     (3) 9b Euler doit s'ecarter de (1+r) de l'ordre de fb/lambda par pas.
#   Un etage A ne verifie qu'une chose. L'etage C en verifie trois, dont deux chiffrees.
HOT10 = {}

def hot10_baseline():
    """Niveau de Hotelling du MEME reseau a cross=False, sur les MEMES dates figees.
       C'est le point zero de STEP 10. Sans lui, (niveau - mu)/mu additionne le
       deplacement cherche et le biais de niveau du reseau, qui sont du meme ordre."""
    for p_ in ("runs_m3.csv", os.path.expanduser("~/runs_m3.csv"), "/tmp/runs_m3.csv"):
        try:
            import pandas as pd
            d = pd.read_csv(p_)
            # La base doit correspondre au run juge sur TOUT sauf cross : sinon on
            # soustrait le biais d'un autre reseau. On ne filtre que sur les colonnes
            # presentes, pour rester compatible avec les lignes anterieures du CSV.
            m_ = d["cross"].astype(str).str.lower() == "false"
            for k_, v_ in (("costs", CFG["costs"]), ("kappas", CFG["kappas"]),
                           ("stocks", CFG["stocks"]), ("blocs", CFG["blocs"]),
                           ("T", CFG["T"]), ("soff_mode", CFG["soff_mode"]),
                           ("env_hard", CFG["env_hard"]), ("dfoc", CFG["dfoc"])):
                if k_ in d.columns:
                    m_ = m_ & (d[k_].astype(str).str.lower() == str(v_).lower())
            d = d[m_]
            r_ = d.iloc[-1]
            b = {i: float(r_[f"hot10_a{i}"]) for i in STRAT_IDX
                 if f"hot10_a{i}" in d.columns and r_[f"hot10_a{i}"] == r_[f"hot10_a{i}"]}
            if b: return b, p_, str(r_.get("run_id", "?"))
        except Exception:
            continue
    return {}, None, None

# ESTIMAND FIXE : dates interieures de l'ORACLE intersectees avec les dates CONJOINTEMENT
# interieures. Calcule DANS TOUS LES CAS, y compris cross=False : c'est ce niveau-la que
# le run etage C suivant relira comme point zero, via runs_m3.csv.
_need10 = 2 if len(STRAT_IDX) >= 2 else 1
_joint10 = INT[:, STRAT_IDX].sum(1) >= _need10
for i in STRAT_IDX:
    m10 = INT[:, i] & _joint10
    if m10.sum() > 1:
        HOT10[i] = (float(hall[m10, i].mean()), int(m10.sum()))

if not CFG["cross"]:
    print("\n" + "="*70)
    print("STEP 10  LIGNE DE BASE (cross=False) : POINT ZERO pour l'etage C")
    print("="*70)
    for i in STRAT_IDX:
        if i in HOT10:
            print(f"    a{i} : niveau {HOT10[i][0]:.4f} sur {HOT10[i][1]} dates figees"
                  f"   ({HOT10[i][0]/mu_a[i]-1:+.2%} de mu = BIAIS du reseau, sans feedback)")
        else:
            print(f"    a{i} : pas de date conjointement interieure, non identifie")
    print("  Ce niveau est logue en CSV (hot10_a*). Le prochain run cross=True le relira")
    print("  et imprimera le DEPLACEMENT VRAI = (niveau_C - niveau_base)/mu.")
    print("="*70)

if CFG["cross"]:
    print("\n" + "="*70)
    print("STEP 10  ETAGE C : ECART MESURE vs PREDICTION INDEPENDANTE")
    print("="*70)
    # ESTIMAND : le niveau de Hotelling DEFLATE sur les dates INTERIEURES, seul endroit
    # ou la rente est identifiee. NE PAS lire lambda(S0,0) : a t=0 les strategiques sont a
    # kappa dans cette calibration, le chiffre serait un artefact.
    # LE JEU DE DATES EST FIGE. Le masque des dates interieures etait calcule sur le rollout du
    # RESEAU, donc il changeait d'un run a l'autre (30 dates a l'etage A2, 38 a l'etage C
    # pour a1) : une partie du deplacement mesure etait un effet de COMPOSITION
    # d'echantillon et non un effet economique. On fige le jeu de dates sur l'ORACLE,
    # intersecte avec les dates conjointement interieures. Le NIVEAU reste mesure sur le
    # reseau ; seul le domaine d'evaluation est gele.
    print("  deplacement de la rente sur l'ESTIMAND FIXE (dates interieures de l'oracle")
    print("  intersectees avec les dates conjointement interieures) :")
    for i in STRAT_IDX:
        if i not in HOT10:
            print(f"    a{i} : pas de date conjointement interieure, non identifie"); continue
        obs = HOT10[i][0]/mu_a[i] - 1.0
        pre = FB_REF[i]
        ok = ("OK" if (abs(pre) > 1e-9 and abs(obs-pre) < 0.5*abs(pre))
              else ("signe OK, amplitude hors tolerance" if obs*pre > 0 else "SIGNE FAUX"))
        print(f"    a{i} : observe {obs:+.2%}  |  predit {pre:+.2%}  |  {ok}"
              f"   ({HOT10[i][1]} dates figees)")
    # controle de composition : de combien le seul changement de jeu de dates deplace-t-il
    # le chiffre ? Si c'est du meme ordre que fb_ref, l'ancien estimand ne mesurait rien.
    for i in STRAT_IDX:
        if i in HOT and i in HOT10:
            print(f"    a{i} : ancien estimand (masque rollout) "
                  f"{HOT[i][1]/mu_a[i]-1.0:+.2%} sur {HOT[i][2]} dates "
                  f"-> effet de composition {abs(HOT10[i][0]-HOT[i][1])/mu_a[i]:.2%}")

    # LE JUGE : la difference avec le meme reseau sans injection. L'ecart a mu ci-dessus
    # contient le biais de niveau du reseau (+0.8% sur a0 a l'etage A2), du meme ordre que
    # fb_ref. Seule la difference des deux runs isole le deplacement d'equilibre.
    _base, _bp, _brid = hot10_baseline()
    if _base:
        print(f"\n  POINT ZERO : run {_brid} (cross=False) lu dans {_bp}")
        for i in STRAT_IDX:
            if i not in _base or i not in HOT10: continue
            dv = (HOT10[i][0] - _base[i])/mu_a[i]
            ok = ("OK" if (abs(FB_REF[i]) > 1e-9 and abs(dv-FB_REF[i]) < 0.5*abs(FB_REF[i]))
                  else ("signe OK, amplitude hors tolerance" if dv*FB_REF[i] > 0
                        else "SIGNE FAUX"))
            print(f"    a{i} : DEPLACEMENT VRAI {dv:+.2%}  |  predit {FB_REF[i]:+.2%}  |  {ok}"
                  f"   (biais de base {_base[i]/mu_a[i]-1:+.2%})")
    else:
        print("\n  AUCUN point zero cross=False pour cette calibration dans le CSV.")
        print("  Les ecarts a mu ci-dessus MELANGENT le deplacement et le biais de niveau")
        print("  du reseau : ils ne mesurent PAS fb_ref. Lancer la ligne de base d'abord.")
    print("  fb_ref est une prediction du PREMIER ORDRE : le feedback evalue A LA SOLUTION")
    print("  OPEN-LOOP. L'equilibre markovien deplace le sentier, donc les termes d'ordre")
    print("  superieur ne sont pas captures et l'egalite exacte n'est PAS attendue. Le juge")
    print("  irrefutable reste 9d MARKOV ; ceci est un controle de signe et d'amplitude.")
    if "dev_markov" in STEP9:
        print(f"\n  deviation MARKOV {STEP9['dev_markov']:.2e} (doit etre ~0)"
              f"  |  FREEZE {STEP9['dev_freeze']:.2e} (doit MONTER vs etage A ~5e-5)")
        _rt = STEP9["dev_freeze"]/max(STEP9["dev_markov"], 1e-12)
        if _rt > 3.0:
            print(f"    -> BASCULE CONFIRMEE (ratio {_rt:.2f}) : le profil markovien est un")
            print("       equilibre, l'open-loop non.")
        elif STEP9["dev_markov"] < STEP9["dev_freeze"]:
            print(f"    -> NON CONCLUANT (ratio {_rt:.2f} < 3) : l'ordre est bon mais les deux")
            print("       gains sont au plancher de bruit. Un signe ne suffit pas.")
        else:
            print("    -> ORDRE INVERSE : le reseau n'a pas bascule en closed-loop. Verifier")
            print("       que feedback est non nul (STEP 8) et que cross=True a bien pris.")
    print("="*70)


# figures
if CFG["make_figs"]:
    try:
        import matplotlib.pyplot as plt
        os.makedirs("figs", exist_ok=True); tag = CFG["tag"]
        fig, axs = plt.subplots(NAG+2, 1, figsize=(6.6, 2.0*(NAG+2)), sharex=True)
        for i in range(NAG):
            axs[i].plot(xa[:, i], "k-", lw=2, label="oracle open-loop")
            axs[i].plot(X[:, i], "r--", lw=1.4, label="reseau")
            if MPE: axs[i].plot(Xcl[:, i], "g-.", lw=1.2, label="grille MPE")
            axs[i].axhline(KAP_np[i], color="gray", ls=":", lw=1)
            axs[i].set_ylabel(f"$x_{i}$ (bloc {BLOC[i]})"); axs[i].legend(fontsize=7)
        axs[NAG].plot(pa, "k-", lw=2); axs[NAG].plot(P, "r--", lw=1.4)
        if MPE: axs[NAG].plot(Pcl, "g-.", lw=1.2)
        axs[NAG].set_ylabel("$p$")
        axs[NAG+1].semilogy(ia, np.maximum(ep, 1e-8), "b.-")
        axs[NAG+1].set_ylabel("err rel. $p$"); axs[NAG+1].set_xlabel("t")
        fig.tight_layout(); fig.savefig(f"figs/{tag}_rollout.png", dpi=160)
        np.savez(f"figs/{tag}_curves.npz", loss=np.array(loss_hist), X=X, P=P, xa=xa, pa=pa,
                 err_p=err_p, A=A_np, mu=mu_a)
        plt.show()
        print(f"\nfigures -> figs/{tag}_rollout.png (+ .npz)")
    except Exception as e:
        print(f"\n[figures ignorees : {e}]")


# log CSV et verdict
row = dict(CFG)
row.update(run_id=time.strftime("%Y%m%d-%H%M%S"), secs=round(time.time()-T_START), nag=NAG,
           fb_h_spread=FBH_SPREAD,
           loss=LOSS_FINAL, err_p=err_p, err_x=err_x,
           err_p_noext=ERRP_NOEXT, err_p_q90=ERRP_Q90,
           lam_path_med=LAM_MED, lam_path_max=LAM_MAX,
           field_int_med=FI[0], field_int_q90=FI[1],
           rec_08=REC.get(0.8), rec_12=REC.get(1.2),
           bellman_gap=BELL, coh=COH, cross_neg=CROSS_NEG, feedback_rel=FB,
           feedback_neg_frac=FB_SIGN,
           fb_cum_max=(float(np.nanmax(np.abs(FBSCR[STRAT_IDX]))) if STRAT_IDX else np.nan),
           lam0_shift=json.dumps([float(round(v, 5)) for v in
                                  (FBSCR if not CFG["cross"] else FB_REF)]),
           mpe_ng=MPE.get("ng"), mpe_br_resid=MPE.get("br_resid"),
           mpe_mono_err_p=MPE.get("mono_err_p"),
           mpe_gap_p=MPE.get("gap_p"), mpe_gap_p_noext=MPE.get("gap_p_noext"),
           mpe_gap_p_q90=MPE.get("gap_p_q90"), mpe_gap_p0=MPE.get("gap_p0"),
           mu_oracle=json.dumps(list(np.round(mu_a, 4))))
row.update({f"s9_{k}": v for k, v in STEP9.items()})
row.update({f"sw_lam_a{i}": v[0] for i, v in SWITCH.items()})
row.update({f"sw_dv_a{i}":  v[1] for i, v in SWITCH.items()})
row.update({f"dec_t{k}": v for k, v in DEC.items()})
row.update({f"hot_cv_a{i}": v[0] for i, v in HOT.items()})
row.update({f"hot10_a{i}": v[0] for i, v in HOT10.items()})
row.update({f"hot10_n_a{i}": v[1] for i, v in HOT10.items()})
# Un run de 2h perdu sur un Permission denied ne doit plus arriver : on tente plusieurs
# chemins, et on depose de toute facon la ligne brute en JSON quelque part d'accessible.
CSV_PATH = None
for p_csv in ("runs_m3.csv", os.path.expanduser("~/runs_m3.csv"), "/tmp/runs_m3.csv"):
    try:
        import pandas as pd
        df = pd.DataFrame([row])
        if os.path.exists(p_csv): df = pd.concat([pd.read_csv(p_csv), df], ignore_index=True)
        df.to_csv(p_csv, index=False); CSV_PATH = p_csv
        print(f"[E] run logue dans {p_csv} ({len(df)} lignes)")
        break
    except Exception as e:
        print(f"[E] {p_csv} indisponible ({e})")
for _d in ([os.path.dirname(os.path.abspath(CSV_PATH))] if CSV_PATH else []) + [".", "/tmp"]:
    try:
        _jp = os.path.join(_d, f"run_{row['run_id']}.json")
        with open(_jp, "w") as _f: json.dump(row, _f, default=str)
        print(f"[E] ligne brute -> {_jp}")
        break
    except Exception:
        continue
if CSV_PATH is None:
    print("[E] AUCUN CSV ecrit ; ligne brute :\n" + json.dumps(row, default=str))

if not CFG["cross"]:
    # etages A et B : l'oracle est la VERITE. Verdict imprime, pas d'assert : STEP 9 est
    # deja passe et ses chiffres ne doivent pas etre perdus sur un run sous-entraine.
    if err_p >= 5e-2:
        print(f"\n[!] ECHEC : rollout tres loin de l'oracle open-loop ({err_p:.2e}). "
              f"Run sous-entraine ou regression.")
    else:
        print("\nVALIDATION OK" if err_p < 1e-2 else f"\ntolerance large ; err_p={err_p:.2e} > 1%")
else:
    print(f"\nETAGE C (MPE) : ecart a l'open-loop = {err_p:.2%} sur le prix. "
          f"Ce n'est pas une erreur, c'est la mesure de la difference entre les deux concepts.")
    print(f"  Le juge est STEP 9d : le gain de deviation MARKOVIEN doit etre nul.")
    if MPE:
        print(f"  Controle croise grille : ecart CL/OL = {MPE['gap_p_noext']:.2%} "
              f"(plancher numerique mono-agent {MPE['mono_err_p']:.2%}).")

