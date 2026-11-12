"""Figures du rapport (exploration + résultats), générées à partir des fichiers déjà produits.

    python main.py figures

Chaque figure est indépendante : si un fichier source manque (par exemple l'expérience
de robustesse n'a pas été lancée), elle est simplement sautée avec un message.
Toutes les figures sont enregistrées en PNG dans resultats/figures/rapport/.

Choix graphiques : une seule teinte (bleu) quand il n'y a qu'une série ; bleu et orange
pour comparer deux séries ; jamais de double axe ; légende dès qu'il y a deux séries.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import config

BLEU, ORANGE, VERT = "#2a78d6", "#eb6834", "#1baf7a"
TEXTE, TEXTE2, GRILLE, FOND = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "figure.facecolor": FOND, "axes.facecolor": FOND, "savefig.facecolor": FOND,
    "axes.edgecolor": GRILLE, "axes.labelcolor": TEXTE2, "xtick.color": TEXTE2,
    "ytick.color": TEXTE2, "text.color": TEXTE, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRILLE, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 9, "legend.frameon": False,
})


def _sauver(fig, dossier: Path, nom: str) -> Path:
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / f"{nom}.png"
    fig.tight_layout()
    fig.savefig(chemin, dpi=160)
    plt.close(fig)
    print(f"[figures] {chemin}")
    return chemin


def _barres_groupees(ax, categories, series: dict, horizontal=True, log=False):
    """Barres groupées (2 séries max : bleu puis orange), 2 px d'écart entre barres."""
    couleurs = [BLEU, ORANGE]
    n = len(series)
    h = 0.8 / n
    pos = np.arange(len(categories))
    for i, (nom, valeurs) in enumerate(series.items()):
        decalage = pos - 0.4 + h * (i + 0.5)
        if horizontal:
            ax.barh(decalage, valeurs, height=h * 0.9, color=couleurs[i], label=nom)
        else:
            ax.bar(decalage, valeurs, width=h * 0.9, color=couleurs[i], label=nom)
    if horizontal:
        ax.set_yticks(pos, categories)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        if log:
            ax.set_xscale("log")
    else:
        ax.set_xticks(pos, categories, rotation=40, ha="right")
        ax.grid(axis="x", visible=False)
    if n > 1:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=n)


# ---------------------------------------------------------------- exploration
def classes_avant_apres(bilan: Path, sortie: Path):
    """Nombre de flux par classe dans les CSV bruts et après nettoyage (échelle log)."""
    if not bilan.exists():
        return print(f"[figures] sauté : {bilan} introuvable")
    t = pd.read_csv(bilan, index_col=0)
    col_avant = "brutes" if "brutes" in t else "avant"
    t = t.sort_values(col_avant, ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    _barres_groupees(ax, t.index, {"CSV bruts": t[col_avant].clip(lower=1),
                                   "après nettoyage": t["apres"].clip(lower=1)}, log=True)
    ax.set_xlabel("nombre de flux (échelle logarithmique)")
    ax.set_title("Flux par classe, avant et après nettoyage", pad=24)
    return _sauver(fig, sortie, "01_classes_avant_apres")


def doublons_par_classe(bilan: Path, sortie: Path):
    if not bilan.exists() or "uniques" not in pd.read_csv(bilan, nrows=1).columns:
        return print("[figures] sauté : bilan_classes.csv sans colonne « uniques »")
    t = pd.read_csv(bilan, index_col=0)
    pct = (100 * (1 - t["uniques"] / t["brutes"])).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(pct.index, pct.values, color=BLEU, height=0.7)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(pct.values):
        if v >= 40:  # étiquettes seulement pour les cas marquants
            ax.text(v + 1, y, f"{v:.1f} %", va="center", color=TEXTE, fontsize=8)
    ax.set_xlim(0, 110)
    ax.set_xlabel("part des flux qui sont des doublons (%)")
    ax.set_title("Doublons par classe dans les CSV bruts")
    return _sauver(fig, sortie, "02_doublons_par_classe")


def boites_par_classe(df: pd.DataFrame, sortie: Path,
                      colonnes=("Flow Duration", "Flow Pkts/s", "Tot Fwd Pkts", "Fwd Seg Size Min")):
    chemins = []
    ordre = df[config.CIBLE].value_counts().index.tolist()
    for col in colonnes:
        if col not in df:
            continue
        donnees = [df.loc[df[config.CIBLE] == c, col].to_numpy() for c in ordre]
        log = df[col].max() > 1000  # échelle log seulement pour les grandeurs très étalées
        if log:
            donnees = [np.sign(d) * np.log10(1 + np.abs(d)) for d in donnees]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(donnees, vert=False, showfliers=False, widths=0.6, patch_artist=True,
                   boxprops={"facecolor": "#b7d3f6", "edgecolor": BLEU},
                   medianprops={"color": "#0d366b", "linewidth": 1.5},
                   whiskerprops={"color": BLEU}, capprops={"color": BLEU})
        ax.set_yticks(range(1, len(ordre) + 1), ordre)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"log10(1 + {col})" if log else col)
        ax.set_title(f"{col} par classe (sans les points extrêmes)")
        nom = "03_boite_" + col.replace("/", "_").replace(" ", "_")
        chemins.append(_sauver(fig, sortie, nom))
    return chemins


def correlations(df: pd.DataFrame, sortie: Path, n_max=200_000):
    num = [c for c in df.columns if c not in (config.CIBLE, "Jour")
           and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
    x = df[num].sample(min(n_max, len(df)), random_state=config.GRAINE)
    corr = x.corr()
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr.values, cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(num)), num, rotation=90, fontsize=5)
    ax.set_yticks(range(len(num)), num, fontsize=5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="corrélation de Pearson")
    ax.set_title("Corrélations entre caractéristiques (après nettoyage)")
    return _sauver(fig, sortie, "04_correlations")


def pca_2d(df: pd.DataFrame, sortie: Path, n_par_classe=3000):
    """Projection PCA : Benign, Infilteration et toutes les autres attaques.
    Trois groupes seulement, pour rester lisible et montrer le recouvrement clé."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    num = [c for c in df.columns if c not in (config.CIBLE, "Jour")
           and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
    ech = df.groupby(config.CIBLE, group_keys=False).head(n_par_classe)
    X = ech[num].to_numpy(dtype="float64")
    X = np.sign(X) * np.log1p(np.abs(X))
    Z = PCA(n_components=2, random_state=config.GRAINE).fit_transform(StandardScaler().fit_transform(X))
    groupe = np.where(ech[config.CIBLE] == "Benign", "Benign",
                      np.where(ech[config.CIBLE] == "Infilteration", "Infilteration", "Autres attaques"))
    fig, ax = plt.subplots(figsize=(8, 6))
    for nom, coul in [("Autres attaques", VERT), ("Benign", BLEU), ("Infilteration", ORANGE)]:
        m = groupe == nom
        ax.scatter(Z[m, 0], Z[m, 1], s=9, color=coul, alpha=0.55, label=nom,
                   edgecolors=FOND, linewidths=0.3)
    ax.set_xlabel("composante principale 1")
    ax.set_ylabel("composante principale 2")
    ax.legend(markerscale=2, loc="upper right")
    ax.set_title("Projection PCA : Benign et Infilteration se superposent")
    return _sauver(fig, sortie, "05_pca_benign_infilteration")


# ---------------------------------------------------------------- résultats
def comparaison_modeles(resultats: Path, sortie: Path):
    if not resultats.exists():
        return print(f"[figures] sauté : {resultats} introuvable")
    r = pd.read_csv(resultats)
    r = r[r.ensemble == "test"].sort_values("f1_macro", ascending=False).drop_duplicates("modele")
    r = r[~r.modele.str.contains("sans_|binaire", regex=True)].sort_values("f1_macro")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    couleurs = [BLEU if i == len(r) - 1 else "#86b6ef" for i in range(len(r))]
    ax.barh(r.modele, r.f1_macro, color=couleurs, height=0.65)
    for y, v in enumerate(r.f1_macro):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=8, color=TEXTE)
    ax.set_xlim(0.5, 1.0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("F1 macro sur le jeu de test")
    ax.set_title("Comparaison des modèles")
    return _sauver(fig, sortie, "06_comparaison_modeles")


def f1_par_classe(dossier_resultats: Path, sortie: Path,
                  modeles=("regression_logistique", "mlp", "lightgbm", "random_forest", "xgboost")):
    lignes = {}
    for m in modeles:
        f = dossier_resultats / f"rapport_{m}_test.csv"
        if f.exists():
            t = pd.read_csv(f, index_col=0)
            lignes[m] = t.loc[~t.index.isin(["accuracy", "macro avg", "weighted avg"]), "f1-score"]
    if not lignes:
        return print("[figures] sauté : aucun rapport_<modele>_test.csv")
    t = pd.DataFrame(lignes)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(t.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(t.shape[1]), t.columns, rotation=30, ha="right")
    ax.set_yticks(range(t.shape[0]), t.index)
    ax.grid(False)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.6 else TEXTE)
    fig.colorbar(im, ax=ax, fraction=0.04, label="F1 par classe")
    ax.set_title("F1 par classe et par modèle (test)")
    return _sauver(fig, sortie, "07_f1_par_classe")


def ablation_fenetre(dossier_resultats: Path, sortie: Path,
                     complet="xgboost", sans="xgboost_sans_fenetres"):
    fa, fb = dossier_resultats / f"rapport_{complet}_test.csv", dossier_resultats / f"rapport_{sans}_test.csv"
    if not (fa.exists() and fb.exists()):
        return print("[figures] sauté : rapports de l'ablation introuvables")
    a, b = pd.read_csv(fa, index_col=0), pd.read_csv(fb, index_col=0)
    classes = [c for c in a.index if c not in ("accuracy", "macro avg", "weighted avg")]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _barres_groupees(ax, classes, {"modèle complet": a.loc[classes, "f1-score"],
                                   "sans fenêtre TCP": b.loc[classes, "f1-score"]})
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 sur le jeu de test")
    ax.set_title("Ablation : seules les attaques web dépendent de la fenêtre TCP", pad=24)
    return _sauver(fig, sortie, "08_ablation_fenetre")


def robustesse_jours(fichier: Path, sortie: Path):
    if not fichier.exists():
        return print(f"[figures] sauté : {fichier} introuvable")
    t = pd.read_csv(fichier)
    t["jour"] = t.jour_exclu.str.extract(r"(\d{2}-\d{2})-\d{4}")[0].fillna(t.jour_exclu)
    variantes = [v for v in ["complet", "sans_fenetres"] if v in set(t.variante)][:2]
    noms = {"complet": "modèle complet", "sans_fenetres": "sans fenêtre TCP"}
    ordre = sorted(t.jour.unique(), key=lambda j: (j[3:5], j[:2]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.5))
    for ax, mesure, titre in [(axes[0], "rappel_attaques", "Rappel des attaques"),
                              (axes[1], "taux_fausses_alertes", "Taux de fausses alertes")]:
        series = {noms[v]: t[t.variante == v].set_index("jour").loc[ordre, mesure] for v in variantes}
        _barres_groupees(ax, ordre, series)
        ax.set_xlim(0, 1.05)
        ax.set_title(titre, pad=24)
    axes[1].set_yticklabels([])
    axes[0].set_ylabel("jour de capture exclu de l'entraînement (jj-mm)")
    fig.suptitle("Test sur des jours jamais vus (détection binaire)", x=0.01, ha="left",
                 fontweight="bold")
    return _sauver(fig, sortie, "09_robustesse_jours")


def tout(dossier_donnees: Path, dossier_resultats: Path) -> None:
    sortie = dossier_resultats / "figures" / "rapport"
    bilan = dossier_donnees / "clean" / "bilan_classes.csv"
    classes_avant_apres(bilan, sortie)
    doublons_par_classe(bilan, sortie)
    propre = dossier_donnees / "clean" / "donnees_propres.parquet"
    if propre.exists():
        df = pd.read_parquet(propre)
        boites_par_classe(df, sortie)
        correlations(df, sortie)
        pca_2d(df, sortie)
        del df
    else:
        print(f"[figures] sauté (exploration) : {propre} introuvable")
    comparaison_modeles(dossier_resultats / "resultats.csv", sortie)
    f1_par_classe(dossier_resultats, sortie)
    ablation_fenetre(dossier_resultats, sortie)
    robustesse_jours(dossier_resultats / "robustesse" / "jours_exclus_binaire.csv", sortie)
