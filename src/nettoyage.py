"""Étape 3 : NETTOYAGE, décidé à partir de l'audit (src/audit.py).

Chaque étape est tracée dans un journal : lignes avant / après, colonnes retirées,
et pertes par classe. Ce journal va directement dans le rapport pour justifier
chaque choix (« on a supprimé 0,4 % des lignes à cause des inf, dont 0 attaque web »).

Ordre des étapes, et pourquoi :
  1. Colonnes de fuite      Timestamp, IP, ports source : le modèle apprendrait l'horaire
                            ou la machine, pas le comportement de l'attaque.
  2. inf -> NaN             Flow Byts/s et Flow Pkts/s divisent par une durée nulle.
  3. Lignes avec NaN        Supprimées : l'audit montre qu'elles sont < 1 %. (Alternative :
                            imputation par la médiane, à faire APRÈS le split, sur le train.)
  4. Négatifs impossibles   Durée ou taille < 0 = erreur de mesure. Init Fwd/Bwd Win Byts = -1
                            est conservé : c'est un code « fenêtre non observée ».
  5. Doublons               Sur caractéristiques + Label. À faire AVANT le split, sinon une même
                            ligne peut tomber dans le train ET le test et gonfler les scores.
  6. Conflits d'étiquettes  Caractéristiques identiques, étiquettes différentes : retirés.
  7. Colonnes constantes    Variance nulle = aucune information.
  8. Colonnes dupliquées    Strictement égales à une autre colonne.
  9. (option) Web Attack    Regroupe les 3 attaques web très rares.
 10. Classes trop rares     Moins de `min_par_classe` flux après nettoyage (50 par défaut) :
                            retirées. Dans CSE-CIC-IDS2018, FTP-BruteForce et DoS-SlowHTTPTest
                            sont des doublons à 99,97 % et en conflit avec SSH-Bruteforce :
                            il ne reste qu'une poignée de flux distincts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .audit import caracteristiques, colonnes_dupliquees, colonnes_numeriques


class Journal:
    def __init__(self, df: pd.DataFrame):
        self.etapes = []
        self.classes_depart = df[config.CIBLE].astype(str).value_counts()
        self._classes_avant = self.classes_depart.copy()
        self.chargement = None

    def noter(self, etape: str, avant: int, df: pd.DataFrame, colonnes_retirees=(), detail=""):
        classes = df[config.CIBLE].astype(str).value_counts()
        pertes = (self._classes_avant - classes.reindex(self._classes_avant.index, fill_value=0))
        pertes = pertes[pertes > 0]
        self.etapes.append({
            "etape": etape, "lignes_avant": avant, "lignes_apres": len(df),
            "lignes_retirees": avant - len(df),
            "pct_retire": round(100 * (avant - len(df)) / max(avant, 1), 3),
            "colonnes_retirees": ", ".join(colonnes_retirees),
            "classes_touchees": ", ".join(f"{k}: -{v}" for k, v in pertes.items()),
            "detail": detail,
        })
        self._classes_avant = classes
        print(f"[nettoyage] {etape:<28} {avant:>10,} -> {len(df):>10,} lignes"
              + (f" | colonnes retirées : {', '.join(colonnes_retirees)}" if colonnes_retirees else ""))

    def tableau(self) -> pd.DataFrame:
        return pd.DataFrame(self.etapes)

    def bilan_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        apres = df[config.CIBLE].astype(str).value_counts()
        t = pd.DataFrame({"avant": self.classes_depart,
                          "apres": apres.reindex(self.classes_depart.index, fill_value=0)})
        t["retirees"] = t.avant - t.apres
        t["pct_retire"] = (100 * t.retirees / t.avant).round(2)
        if self.chargement is not None:  # colonnes du chargement économe
            c = self.chargement.reindex(t.index)
            t.insert(0, "uniques", c["uniques"].fillna(0).astype(int))
            t.insert(0, "brutes", c["brutes"].fillna(0).astype(int))
            t = t.rename(columns={"avant": "apres_tirage"})
        return t


def nettoyer(df: pd.DataFrame,
             supprimer_conflits: bool = True,
             regrouper_web: bool = False,
             garder_jour: bool = True,
             min_par_classe: int = 50) -> tuple[pd.DataFrame, Journal]:
    """Applique toutes les étapes et renvoie (données propres, journal)."""
    info = df.attrs.get("chargement")
    j = Journal(df)
    if info:  # chargement économe : les doublons ont déjà été retirés à la lecture
        j.chargement = info["par_classe"]
        j.etapes.append({
            "etape": "0. doublons (au chargement)", "lignes_avant": info["lignes_brutes"],
            "lignes_apres": info["lignes_uniques"], "lignes_retirees": info["doublons_retires"],
            "pct_retire": round(100 * info["doublons_retires"] / info["lignes_brutes"], 3),
            "colonnes_retirees": ", ".join(config.COLONNES_FUITE), "classes_touchees": "",
            "detail": "doublons sur caractéristiques + Label, avant échantillonnage"})
        j.etapes.append({
            "etape": "0b. échantillonnage", "lignes_avant": info["lignes_uniques"],
            "lignes_apres": len(df), "lignes_retirees": info["lignes_uniques"] - len(df),
            "pct_retire": round(100 * (info["lignes_uniques"] - len(df)) / info["lignes_uniques"], 3),
            "colonnes_retirees": "", "classes_touchees": "", "detail": "plafond par classe"})
        print(f"[nettoyage] 0. doublons au chargement : -{info['doublons_retires']:,} lignes")

    # 1. colonnes de fuite
    n = len(df)
    fuite = [c for c in config.COLONNES_FUITE if c in df.columns]
    if not garder_jour and "Jour" in df.columns:
        fuite.append("Jour")
    df = df.drop(columns=fuite)
    j.noter("1. colonnes de fuite", n, df, fuite)

    cols = colonnes_numeriques(df)

    # 2. inf -> NaN
    n = len(df)
    nb_inf = int(np.isinf(df[cols]).sum().sum())
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    j.noter("2. inf -> NaN", n, df, detail=f"{nb_inf} valeurs infinies remplacées")

    # 3. lignes avec NaN
    n = len(df)
    df = df.dropna(subset=cols)
    j.noter("3. lignes avec NaN", n, df)

    # 4. valeurs négatives impossibles
    n = len(df)
    a_controler = [c for c in cols if c not in config.COLONNES_NEG_AUTORISE]
    negatif = (df[a_controler] < 0).any(axis=1)
    df = df.loc[~negatif]
    j.noter("4. négatifs impossibles", n, df)

    # 5. doublons (caractéristiques + Label)
    n = len(df)
    feats = caracteristiques(df)
    df = df.drop_duplicates(subset=feats + [config.CIBLE])
    j.noter("5. doublons", n, df)

    # 6. conflits d'étiquettes
    if supprimer_conflits:
        n = len(df)
        conflit = df.duplicated(subset=feats, keep=False)
        df = df.loc[~conflit]
        j.noter("6. conflits d'étiquettes", n, df)

    # 7. colonnes constantes
    n = len(df)
    constantes = [c for c in colonnes_numeriques(df) if df[c].nunique(dropna=True) <= 1]
    df = df.drop(columns=constantes)
    j.noter("7. colonnes constantes", n, df, constantes)

    # 8. colonnes dupliquées
    n = len(df)
    dup = colonnes_dupliquees(df)
    a_retirer = sorted(set(dup["identique_a"])) if not dup.empty else []
    df = df.drop(columns=a_retirer)
    j.noter("8. colonnes dupliquées", n, df, a_retirer)

    # 9. regroupement des attaques web (optionnel)
    if regrouper_web:
        n = len(df)
        df[config.CIBLE] = df[config.CIBLE].astype(str).replace(
            {c: "Web Attack" for c in config.CLASSES_WEB})
        j.noter("9. regroupement Web Attack", n, df, detail=" + ".join(config.CLASSES_WEB))

    # 10. classes devenues trop rares pour être apprises (et pour un split stratifié)
    if min_par_classe:
        n = len(df)
        comptes = df[config.CIBLE].astype(str).value_counts()
        rares = sorted(comptes[comptes < min_par_classe].index)
        df = df.loc[~df[config.CIBLE].astype(str).isin(rares)]
        j.noter("10. classes trop rares", n, df,
                detail=f"moins de {min_par_classe} flux après nettoyage : {', '.join(rares) or 'aucune'}")

    df[config.CIBLE] = df[config.CIBLE].astype(str)
    df = df.reset_index(drop=True)
    df.attrs = {}  # infos de chargement déjà copiées dans le journal
    return df, j


def sauvegarder(df: pd.DataFrame, journal: Journal,
                dossier: Path = config.DOSSIER_PROPRE) -> Path:
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / "donnees_propres.parquet"
    df.to_parquet(chemin, index=False)
    journal.tableau().to_csv(dossier / "journal_nettoyage.csv", index=False)
    journal.bilan_classes(df).to_csv(dossier / "bilan_classes.csv")
    print(f"[nettoyage] {len(df):,} lignes, {df.shape[1]} colonnes -> {chemin}")
    return chemin


def charger_propre(dossier: Path = config.DOSSIER_PROPRE) -> pd.DataFrame:
    return pd.read_parquet(Path(dossier) / "donnees_propres.parquet")
