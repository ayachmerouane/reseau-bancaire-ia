"""Étape 2 : AUDIT de la qualité des données, AVANT tout nettoyage.

Rien n'est modifié ici. Chaque fonction mesure un type de problème et renvoie un
tableau, pour que le binôme décide ensuite, en connaissance de cause, quoi nettoyer
et comment (et puisse le justifier dans le rapport).

Contrôles réalisés :
  1. Vue d'ensemble          : lignes, colonnes, types, mémoire
  2. Défauts de lecture      : en-têtes répétés, textes non numériques, colonnes par fichier
  3. Valeurs manquantes      : NaN par colonne et par classe
  4. Valeurs infinies        : inf / -inf par colonne et par classe
  5. Doublons                : lignes identiques (avec et sans Timestamp/Jour/IP)
  6. Conflits d'étiquettes   : mêmes caractéristiques, étiquettes différentes
  7. Colonnes constantes     : variance nulle ou quasi nulle
  8. Colonnes dupliquées     : deux colonnes exactement égales
  9. Valeurs négatives       : impossibles pour une durée, une taille, un débit
 10. Valeurs aberrantes      : méthode IQR (signalées seulement : une attaque EST souvent extrême)
 11. Domaines de validité    : Protocol ∈ {0, 6, 17}, Dst Port ∈ [0, 65535]
 12. Cible                   : liste des classes, déséquilibre
 13. Corrélations fortes     : paires de colonnes |r| > 0,95

Deux niveaux :
  - audit_complet(dossier_parquet) : comptages EXACTS sur les 16 M de lignes, fichier par
    fichier, sans tout charger en mémoire (NaN, inf, négatifs, classes, doublons).
  - rapport_audit(df) : tous les contrôles, sur un DataFrame (échantillon ou complet).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config
from .chargement import lire_journal, lister_parquet

NON_CARACTERISTIQUES = [config.CIBLE, "Jour"] + config.COLONNES_FUITE


def colonnes_numeriques(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in NON_CARACTERISTIQUES and pd.api.types.is_numeric_dtype(df[c])]


def caracteristiques(df: pd.DataFrame) -> list[str]:
    """Colonnes qui serviront à l'apprentissage (hors cible, horaire, IP, jour)."""
    return [c for c in df.columns if c not in NON_CARACTERISTIQUES]


# ---------------------------------------------------------------- 1. vue d'ensemble
def vue_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "indicateur": ["lignes", "colonnes", "colonnes numériques", "colonnes texte",
                       "mémoire (Mo)"],
        "valeur": [len(df), df.shape[1], len(colonnes_numeriques(df)),
                   int(sum(not pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)),
                   round(df.memory_usage(deep=True).sum() / 1e6, 1)],
    })


def types_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """Type de chaque colonne et si ses valeurs sont en fait entières (compteurs, ports)."""
    lignes = []
    for c in df.columns:
        s = df[c]
        entier = None
        if pd.api.types.is_numeric_dtype(s):
            v = s.replace([np.inf, -np.inf], np.nan).dropna()
            entier = bool(len(v) and np.all(np.mod(v.to_numpy(dtype="float64"), 1) == 0))
        lignes.append({"colonne": c, "type": str(s.dtype), "valeurs_entieres": entier,
                       "nb_valeurs_distinctes": int(s.nunique(dropna=True))})
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------- 2. défauts de lecture
def defauts_lecture(dossier_parquet: Path = config.DOSSIER_PARQUET) -> pd.DataFrame:
    journal = lire_journal(dossier_parquet)
    if not journal:
        return pd.DataFrame()
    toutes = [set(j["colonnes"] or []) for j in journal.values()]
    communes = set.intersection(*toutes) if toutes else set()
    lignes = []
    for jour, j in journal.items():
        en_plus = sorted(set(j["colonnes"] or []) - communes)
        lignes.append({"jour": jour, "lignes_lues": j["lignes_lues"],
                       "entetes_repetes": j["entetes_repetes"],
                       "valeurs_non_numeriques": sum(j["valeurs_non_numeriques"].values()),
                       "nb_colonnes": len(j["colonnes"] or []),
                       "colonnes_en_plus": ", ".join(en_plus)})
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------- 3 et 4. NaN et inf
def valeurs_manquantes(df: pd.DataFrame) -> pd.DataFrame:
    n = df.isna().sum()
    t = pd.DataFrame({"nb_nan": n, "pct_nan": (100 * n / len(df)).round(4)})
    return t[t.nb_nan > 0].sort_values("nb_nan", ascending=False)


def valeurs_infinies(df: pd.DataFrame) -> pd.DataFrame:
    cols = colonnes_numeriques(df)
    pos = np.isposinf(df[cols]).sum()
    neg = np.isneginf(df[cols]).sum()
    t = pd.DataFrame({"nb_+inf": pos, "nb_-inf": neg})
    t["pct"] = (100 * (t["nb_+inf"] + t["nb_-inf"]) / len(df)).round(4)
    return t[(t["nb_+inf"] + t["nb_-inf"]) > 0].sort_values("pct", ascending=False)


def problemes_par_classe(df: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque classe : combien de lignes ont au moins un NaN ou un inf.
    Important : si une classe rare perd beaucoup de lignes au nettoyage, il faut le savoir."""
    cols = colonnes_numeriques(df)
    a_nan = df[cols].isna().any(axis=1)
    a_inf = np.isinf(df[cols]).any(axis=1)
    g = pd.DataFrame({config.CIBLE: df[config.CIBLE].astype(str),
                      "nan": a_nan, "inf": a_inf, "nan_ou_inf": a_nan | a_inf})
    t = g.groupby(config.CIBLE).agg(lignes=("nan", "size"), avec_nan=("nan", "sum"),
                                    avec_inf=("inf", "sum"), avec_nan_ou_inf=("nan_ou_inf", "sum"))
    t["pct_perdu_si_suppression"] = (100 * t.avec_nan_ou_inf / t.lignes).round(3)
    return t.sort_values("lignes", ascending=False)


# ---------------------------------------------------------------- 5. doublons
def doublons(df: pd.DataFrame) -> pd.DataFrame:
    """Trois définitions, de la plus stricte à la plus utile pour l'apprentissage."""
    feats = caracteristiques(df)
    resultats = [
        ("toutes les colonnes identiques", int(df.duplicated().sum())),
        ("caractéristiques + Label identiques (sans Timestamp, IP, Jour)",
         int(df.duplicated(subset=feats + [config.CIBLE]).sum())),
        ("caractéristiques identiques (Label ignoré)", int(df.duplicated(subset=feats).sum())),
    ]
    t = pd.DataFrame(resultats, columns=["definition", "nb_doublons"])
    t["pct"] = (100 * t.nb_doublons / len(df)).round(3)
    return t


def doublons_par_classe(df: pd.DataFrame) -> pd.DataFrame:
    feats = caracteristiques(df)
    d = df.duplicated(subset=feats + [config.CIBLE])
    t = pd.DataFrame({config.CIBLE: df[config.CIBLE].astype(str), "doublon": d})
    t = t.groupby(config.CIBLE).agg(lignes=("doublon", "size"), doublons=("doublon", "sum"))
    t["pct_doublons"] = (100 * t.doublons / t.lignes).round(2)
    t["lignes_uniques"] = t.lignes - t.doublons
    return t.sort_values("lignes", ascending=False)


# ---------------------------------------------------------------- 6. conflits d'étiquettes
def conflits_etiquettes(df: pd.DataFrame) -> pd.DataFrame:
    """Flux aux caractéristiques identiques mais d'étiquettes différentes.
    Aucun modèle ne peut les séparer : c'est une borne sur la performance atteignable."""
    feats = caracteristiques(df)
    cle = pd.util.hash_pandas_object(df[feats], index=False)
    g = pd.DataFrame({"cle": cle.to_numpy(), config.CIBLE: df[config.CIBLE].astype(str).to_numpy()})
    nb = g.groupby("cle")[config.CIBLE].nunique()
    cles_conflit = nb[nb > 1].index
    c = g[g.cle.isin(cles_conflit)]
    if c.empty:
        return pd.DataFrame(columns=["classes_en_conflit", "nb_lignes"])
    paires = c.groupby("cle")[config.CIBLE].apply(lambda s: " / ".join(sorted(set(s))))
    t = c.assign(classes_en_conflit=c.cle.map(paires)).groupby("classes_en_conflit").size()
    return t.rename("nb_lignes").sort_values(ascending=False).reset_index()


# ---------------------------------------------------------------- 7 et 8. colonnes inutiles
def colonnes_constantes(df: pd.DataFrame, seuil_quasi: float = 0.999) -> pd.DataFrame:
    lignes = []
    for c in colonnes_numeriques(df):
        s = df[c]
        nuniq = s.nunique(dropna=True)
        freq = s.value_counts(normalize=True, dropna=True)
        top = float(freq.iloc[0]) if len(freq) else 1.0
        if nuniq <= 1 or top >= seuil_quasi:
            lignes.append({"colonne": c, "nb_valeurs_distinctes": int(nuniq),
                           "valeur_dominante": freq.index[0] if len(freq) else None,
                           "pct_valeur_dominante": round(100 * top, 3),
                           "type": "constante" if nuniq <= 1 else "quasi-constante"})
    return pd.DataFrame(lignes)


def colonnes_dupliquees(df: pd.DataFrame) -> pd.DataFrame:
    """Paires de colonnes strictement identiques (ex. Subflow Fwd Pkts = Tot Fwd Pkts)."""
    cols = colonnes_numeriques(df)
    empreinte = {c: pd.util.hash_pandas_object(df[c], index=False).sum() for c in cols}
    paires = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if empreinte[a] == empreinte[b] and df[a].equals(df[b]):
                paires.append({"colonne": a, "identique_a": b})
    return pd.DataFrame(paires)


# ---------------------------------------------------------------- 9. valeurs négatives
def valeurs_negatives(df: pd.DataFrame) -> pd.DataFrame:
    cols = colonnes_numeriques(df)
    n = (df[cols] < 0).sum()
    t = pd.DataFrame({"nb_negatifs": n, "min": df[cols].min()})
    t["autorise"] = t.index.isin(config.COLONNES_NEG_AUTORISE)
    t["pct"] = (100 * t.nb_negatifs / len(df)).round(4)
    return t[t.nb_negatifs > 0].sort_values("nb_negatifs", ascending=False)


# ---------------------------------------------------------------- 10. valeurs aberrantes
def valeurs_aberrantes(df: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """Méthode IQR : hors de [Q1 - k·IQR, Q3 + k·IQR]. On signale, on ne supprime PAS :
    dans un IDS, les attaques sont précisément des comportements extrêmes."""
    cols = colonnes_numeriques(df)
    x = df[cols].replace([np.inf, -np.inf], np.nan)
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    hors = ((x < q1 - k * iqr) | (x > q3 + k * iqr)).sum()
    t = pd.DataFrame({"nb_aberrants": hors, "pct": (100 * hors / len(df)).round(2),
                      "q1": q1, "q3": q3, "max": x.max()})
    return t[t.nb_aberrants > 0].sort_values("pct", ascending=False)


# ---------------------------------------------------------------- 11. domaines de validité
def domaines_validite(df: pd.DataFrame) -> pd.DataFrame:
    regles = []
    if "Protocol" in df:
        v = df["Protocol"].value_counts(dropna=False)
        regles.append({"controle": "Protocol hors {0, 6, 17}",
                       "nb_lignes": int((~df["Protocol"].isin([0, 6, 17])).sum()),
                       "detail": ", ".join(f"{k}: {n}" for k, n in v.items())})
    if "Dst Port" in df:
        p = df["Dst Port"]
        regles.append({"controle": "Dst Port hors [0, 65535]",
                       "nb_lignes": int(((p < 0) | (p > 65535)).sum()),
                       "detail": f"{p.nunique()} ports distincts"})
    if "Flow Duration" in df:
        regles.append({"controle": "Flow Duration < 0",
                       "nb_lignes": int((df["Flow Duration"] < 0).sum()), "detail": ""})
    return pd.DataFrame(regles)


# ---------------------------------------------------------------- 12. cible
def repartition_classes(df: pd.DataFrame) -> pd.DataFrame:
    v = df[config.CIBLE].astype(str).value_counts()
    t = pd.DataFrame({"nb_flux": v, "pct": (100 * v / v.sum()).round(4)})
    t["ratio_vs_majoritaire"] = (v.iloc[0] / v).round(1)
    return t


# ---------------------------------------------------------------- 13. corrélations
def correlations_fortes(df: pd.DataFrame, seuil: float = 0.95,
                        max_lignes: int = 200_000) -> pd.DataFrame:
    cols = [c for c in colonnes_numeriques(df) if df[c].nunique() > 1]
    x = df[cols].replace([np.inf, -np.inf], np.nan)
    if len(x) > max_lignes:
        x = x.sample(max_lignes, random_state=config.GRAINE)
    corr = x.corr().abs()
    haut = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    t = haut.stack().reset_index()
    t.columns = ["colonne_1", "colonne_2", "abs_correlation"]
    return t[t.abs_correlation > seuil].sort_values("abs_correlation", ascending=False)


# ---------------------------------------------------------------- rapport complet
def rapport_audit(df: pd.DataFrame, dossier_parquet: Path | None = config.DOSSIER_PARQUET,
                  sortie: Path | None = config.DOSSIER_RESULTATS / "audit",
                  afficher: bool = True) -> dict[str, pd.DataFrame]:
    """Lance tous les contrôles, affiche un résumé et enregistre chaque tableau en CSV."""
    r = {
        "01_vue_ensemble": vue_ensemble(df),
        "02_defauts_lecture": defauts_lecture(dossier_parquet) if dossier_parquet else pd.DataFrame(),
        "03_valeurs_manquantes": valeurs_manquantes(df),
        "04_valeurs_infinies": valeurs_infinies(df),
        "05_nan_inf_par_classe": problemes_par_classe(df),
        "06_doublons": doublons(df),
        "07_doublons_par_classe": doublons_par_classe(df),
        "08_conflits_etiquettes": conflits_etiquettes(df),
        "09_colonnes_constantes": colonnes_constantes(df),
        "10_colonnes_dupliquees": colonnes_dupliquees(df),
        "11_valeurs_negatives": valeurs_negatives(df),
        "12_valeurs_aberrantes": valeurs_aberrantes(df),
        "13_domaines_validite": domaines_validite(df),
        "14_repartition_classes": repartition_classes(df),
        "15_correlations_fortes": correlations_fortes(df),
        "16_types_colonnes": types_colonnes(df),
    }
    if sortie is not None:
        Path(sortie).mkdir(parents=True, exist_ok=True)
        for nom, t in r.items():
            t.to_csv(Path(sortie) / f"{nom}.csv", index=not isinstance(t.index, pd.RangeIndex))
    if afficher:
        afficher_resume(r)
    return r


def afficher_resume(r: dict[str, pd.DataFrame]) -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    titres = {
        "01_vue_ensemble": "Vue d'ensemble",
        "02_defauts_lecture": "Défauts de lecture (en-têtes répétés, colonnes par fichier)",
        "03_valeurs_manquantes": "Valeurs manquantes (NaN)",
        "04_valeurs_infinies": "Valeurs infinies",
        "05_nan_inf_par_classe": "NaN / inf par classe",
        "06_doublons": "Doublons",
        "07_doublons_par_classe": "Doublons par classe",
        "08_conflits_etiquettes": "Conflits d'étiquettes",
        "09_colonnes_constantes": "Colonnes constantes ou quasi constantes",
        "10_colonnes_dupliquees": "Colonnes identiques entre elles",
        "11_valeurs_negatives": "Valeurs négatives",
        "12_valeurs_aberrantes": "Valeurs aberrantes (IQR, k=3), 10 premières",
        "13_domaines_validite": "Domaines de validité",
        "14_repartition_classes": "Répartition des classes",
        "15_correlations_fortes": "Corrélations > 0,95, 15 premières",
    }
    for cle, titre in titres.items():
        t = r.get(cle, pd.DataFrame())
        print(f"\n=== {titre} ===")
        if t.empty:
            print("  aucun problème détecté")
        else:
            print(t.head(15 if "correl" in cle else 10 if "aberr" in cle else 50).to_string())


# ---------------------------------------------------------------- audit exact, en flux
def audit_complet(dossier_parquet: Path = config.DOSSIER_PARQUET,
                  sortie: Path | None = config.DOSSIER_RESULTATS / "audit_complet") -> dict:
    """Comptages exacts sur TOUT le jeu, un lot à la fois (quelques centaines de Mo de RAM).

    Les doublons sont détectés grâce à une empreinte (hash 64 bits) de chaque ligne,
    calculée sur les caractéristiques + Label, sans Timestamp ni IP ni Jour.
    """
    classes, nan_col, inf_col, neg_col = {}, {}, {}, {}
    empreintes, lignes_par_jour = [], {}
    total = 0
    for chemin in lister_parquet(dossier_parquet):
        jour = chemin.stem
        lignes_par_jour[jour] = 0
        for lot in pq.ParquetFile(chemin).iter_batches(batch_size=500_000):
            df = lot.to_pandas()
            total += len(df)
            lignes_par_jour[jour] += len(df)
            for k, v in df[config.CIBLE].astype(str).value_counts().items():
                classes[k] = classes.get(k, 0) + int(v)
            cols = colonnes_numeriques(df)
            for k, v in df[cols].isna().sum().items():
                nan_col[k] = nan_col.get(k, 0) + int(v)
            for k, v in np.isinf(df[cols]).sum().items():
                inf_col[k] = inf_col.get(k, 0) + int(v)
            for k, v in (df[cols] < 0).sum().items():
                neg_col[k] = neg_col.get(k, 0) + int(v)
            feats = [c for c in caracteristiques(df) if c not in config.COLONNES_IDENTIFIANTS]
            empreintes.append(pd.util.hash_pandas_object(df[feats + [config.CIBLE]],
                                                         index=False).to_numpy())
        print(f"[audit complet] {jour}: {lignes_par_jour[jour]:,} lignes")
    h = np.concatenate(empreintes)
    nb_uniques = len(np.unique(h))
    res = {
        "total_lignes": pd.DataFrame({"indicateur": ["lignes", "lignes uniques", "doublons", "pct_doublons"],
                                      "valeur": [total, nb_uniques, total - nb_uniques,
                                                 round(100 * (total - nb_uniques) / total, 3)]}),
        "lignes_par_jour": pd.Series(lignes_par_jour, name="lignes").to_frame(),
        "classes": pd.Series(classes, name="nb_flux").sort_values(ascending=False).to_frame(),
        "nan_par_colonne": pd.Series(nan_col, name="nb_nan").loc[lambda s: s > 0].to_frame(),
        "inf_par_colonne": pd.Series(inf_col, name="nb_inf").loc[lambda s: s > 0].to_frame(),
        "negatifs_par_colonne": pd.Series(neg_col, name="nb_negatifs").loc[lambda s: s > 0].to_frame(),
    }
    if sortie is not None:
        Path(sortie).mkdir(parents=True, exist_ok=True)
        for nom, t in res.items():
            t.to_csv(Path(sortie) / f"{nom}.csv")
    for nom, t in res.items():
        print(f"\n=== {nom} ===\n{t.to_string()}")
    return res
