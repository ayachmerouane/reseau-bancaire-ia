# %% [markdown]
# # 01 : Exploration, audit de qualité et nettoyage (CSE-CIC-IDS2018)
#
# **Démarche** : on regarde d'abord TOUS les problèmes possibles, sans rien modifier
# (partie A : audit), on les visualise (partie B : exploration), puis seulement on
# décide et on applique le nettoyage (partie C), avec un journal qui justifie chaque choix.
#
# Prérequis : `python main.py convertir` a déjà transformé les CSV en Parquet.

# %%
import os, sys
from pathlib import Path

RACINE = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RACINE))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from src import config, audit, nettoyage
from src.chargement import charger

DONNEES = Path(os.environ.get("DONNEES", RACINE / "data"))
PARQUET = DONNEES / "parquet"
FIGURES = RACINE / "resultats" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid")
%matplotlib inline

# %% [markdown]
# ## Chargement d'un échantillon de travail
# Les 16 M de lignes ne tiennent pas toutes en mémoire sur un portable. On garde **toutes
# les attaques** et une fraction des flux Benign (`FRAC_BENIGN`). Pour les comptages exacts
# sur le jeu complet, lancer `python main.py audit-complet` (résultats dans `resultats/audit_complet/`).

# %%
FRAC_BENIGN = float(os.environ.get("FRAC_BENIGN", 0.05))
df = charger(PARQUET, frac_benign=FRAC_BENIGN)
print(df.shape)
df.head()

# %% [markdown]
# # Partie A : audit (rien n'est modifié)
# ### A1. Vue d'ensemble et types

# %%
display(audit.vue_ensemble(df))
types = audit.types_colonnes(df)
display(types.head(15))
print("Colonnes texte :", types.loc[~types.type.str.contains("float|int"), "colonne"].tolist())

# %% [markdown]
# ### A2. Défauts de lecture des CSV
# Lignes d'en-tête répétées au milieu des fichiers, valeurs texte dans des colonnes
# numériques, colonnes en plus dans certains fichiers (le 20/02 a `Flow ID`, `Src IP`, `Src Port`, `Dst IP`).

# %%
audit.defauts_lecture(PARQUET)

# %% [markdown]
# ### A3. Valeurs manquantes (NaN)

# %%
nan = audit.valeurs_manquantes(df)
display(nan)

# %% [markdown]
# ### A4. Valeurs infinies
# `Flow Byts/s` et `Flow Pkts/s` divisent par la durée du flux : une durée nulle donne `inf`.

# %%
display(audit.valeurs_infinies(df))

# %% [markdown]
# ### A5. Qui perdrait-on en supprimant les lignes NaN / inf ?
# Si une classe rare est très touchée, la suppression n'est pas acceptable et il faut imputer.

# %%
display(audit.problemes_par_classe(df))

# %% [markdown]
# ### A6. Doublons
# Trois définitions. Celle qui compte pour l'apprentissage est la 2e : mêmes
# caractéristiques et même Label, en ignorant l'horaire et les IP.

# %%
display(audit.doublons(df))
display(audit.doublons_par_classe(df))

# %% [markdown]
# ### A7. Conflits d'étiquettes
# Même vecteur de caractéristiques, étiquettes différentes : aucun modèle ne peut les séparer.

# %%
display(audit.conflits_etiquettes(df))

# %% [markdown]
# ### A8. Colonnes constantes, quasi constantes et colonnes identiques

# %%
display(audit.colonnes_constantes(df))
display(audit.colonnes_dupliquees(df))

# %% [markdown]
# ### A9. Valeurs négatives et domaines de validité
# `Init Fwd Win Byts = -1` est un code (« fenêtre non observée »), pas une erreur.

# %%
display(audit.valeurs_negatives(df))
display(audit.domaines_validite(df))

# %% [markdown]
# ### A10. Valeurs aberrantes (IQR)
# On les **signale** mais on ne les supprime pas : dans un IDS, les attaques sont justement
# des comportements extrêmes (un DoS a un débit anormal).

# %%
audit.valeurs_aberrantes(df).head(15)

# %% [markdown]
# ### Enregistrement de tout l'audit (un CSV par contrôle, pour le rapport)

# %%
_ = audit.rapport_audit(df, PARQUET, sortie=RACINE / "resultats" / "audit", afficher=False)

# %% [markdown]
# # Partie B : exploration visuelle
# ### B1. Répartition des classes (échelle logarithmique)

# %%
rep = audit.repartition_classes(df)
display(rep)
fig, ax = plt.subplots(figsize=(10, 6))
rep["nb_flux"][::-1].plot.barh(ax=ax, color="#2b6cb0", logx=True)
ax.set_xlabel("nombre de flux (échelle log)")
ax.set_title(f"Répartition des classes (Benign échantillonné à {FRAC_BENIGN:.0%})")
plt.tight_layout(); fig.savefig(FIGURES / "repartition_classes.png", dpi=130); plt.show()

# %% [markdown]
# ### B2. Protocoles et ports visés par chaque attaque

# %%
display(pd.crosstab(df[config.CIBLE], df["Protocol"]))
top_ports = (df.groupby(config.CIBLE)["Dst Port"]
               .agg(lambda s: ", ".join(f"{int(p)} ({n})" for p, n in s.value_counts().head(3).items())))
top_ports.to_frame("3 ports les plus visés")

# %% [markdown]
# ### B3. Les attaques se distinguent-elles ? Boîtes à moustaches par classe

# %%
for col in ["Flow Duration", "Flow Pkts/s", "Tot Fwd Pkts", "Init Fwd Win Byts"]:
    if col not in df: continue
    d = df[[col, config.CIBLE]].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=d, x=config.CIBLE, y=col, ax=ax, showfliers=False, color="#90cdf4")
    ax.set_yscale("symlog"); ax.tick_params(axis="x", rotation=75)
    ax.set_title(f"{col} par classe (échelle symlog, sans points extrêmes)")
    plt.tight_layout(); fig.savefig(FIGURES / f"boxplot_{col.replace('/', '_').replace(' ', '_')}.png", dpi=130); plt.show()

# %% [markdown]
# ### B4. Corrélations entre caractéristiques
# Beaucoup de colonnes CICFlowMeter sont redondantes (ex. `Subflow Fwd Pkts` et `Tot Fwd Pkts`).

# %%
num = [c for c in audit.colonnes_numeriques(df) if df[c].nunique() > 1]
corr = df[num].replace([np.inf, -np.inf], np.nan).sample(min(len(df), 100_000), random_state=0).corr()
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax, xticklabels=True, yticklabels=True)
ax.tick_params(labelsize=6); ax.set_title("Matrice de corrélation (Pearson)")
plt.tight_layout(); fig.savefig(FIGURES / "correlations.png", dpi=130); plt.show()
audit.correlations_fortes(df).head(20)

# %% [markdown]
# ### B5. Projection en 2D (PCA) d'un échantillon
# Si les classes forment des nuages séparés, la classification sera facile ;
# si elles se superposent (souvent Infilteration et Benign), elle sera difficile.

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.pretraitement import LogSigne

ech = df.replace([np.inf, -np.inf], np.nan).dropna(subset=num)
ech = ech.groupby(config.CIBLE).head(800)
Z = PCA(n_components=2, random_state=0).fit_transform(
    StandardScaler().fit_transform(LogSigne().transform(ech[num])))
fig, ax = plt.subplots(figsize=(11, 8))
sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=ech[config.CIBLE].to_numpy(), s=8, ax=ax, palette="tab20", linewidth=0)
ax.set_title("PCA 2D (log signé + standardisation)"); ax.legend(fontsize=7, markerscale=2, bbox_to_anchor=(1, 1))
plt.tight_layout(); fig.savefig(FIGURES / "pca_2d.png", dpi=130); plt.show()

# %% [markdown]
# # Partie C : nettoyage
# Décisions, justifiées par l'audit ci-dessus :
# 1. retirer `Timestamp` et les IP (fuite d'information) ;
# 2. et 3. inf -> NaN puis supprimer ces lignes (part faible, voir A5) ;
# 4. supprimer les valeurs négatives impossibles (sauf `Init * Win Byts = -1`) ;
# 5. supprimer les doublons AVANT le découpage train/test ;
# 6. supprimer les conflits d'étiquettes ;
# 7. et 8. retirer les colonnes constantes et identiques.

# %%
propre, journal = nettoyage.nettoyer(df, regrouper_web=False)
display(journal.tableau())
display(journal.bilan_classes(propre))

# %% [markdown]
# Vérification : l'audit relancé sur les données propres ne doit plus rien signaler
# (hors valeurs aberrantes, gardées volontairement).

# %%
display(audit.valeurs_manquantes(propre)); display(audit.valeurs_infinies(propre))
display(audit.doublons(propre)); display(audit.colonnes_constantes(propre))

# %%
nettoyage.sauvegarder(propre, journal, DONNEES / "clean")
