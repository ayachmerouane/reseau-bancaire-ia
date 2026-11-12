"""Étape 4 : PRÉTRAITEMENT (encodage, découpage, normalisation, rééquilibrage).

Règle d'or : tout ce qui « apprend » quelque chose des données (moyenne et écart type
du StandardScaler, voisins de SMOTE…) est ajusté sur le jeu d'ENTRAÎNEMENT seulement.
Le jeu de test garde la distribution réelle du trafic et n'est jamais rééquilibré.

La normalisation n'est pas faite ici mais dans un Pipeline scikit-learn (voir
src/modeles.py) : elle est ainsi ajustée automatiquement sur le train, et le modèle
sauvegardé contient son propre scaler (pratique pour la démo).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from . import config
from .audit import caracteristiques

PROTOCOLES = [0, 6, 17]  # autre / TCP / UDP


def encoder_protocol(X: pd.DataFrame) -> pd.DataFrame:
    """Protocol est un code (6 = TCP, 17 = UDP, 0 = autre), pas une quantité :
    on le remplace par 3 colonnes binaires (one-hot). Sans cela, un modèle linéaire
    croirait que UDP (17) « vaut » presque 3 fois TCP (6)."""
    if "Protocol" not in X.columns:
        return X
    X = X.copy()
    for p in PROTOCOLES:
        X[f"Protocol_{p}"] = (X["Protocol"] == p).astype("float32")
    return X.drop(columns=["Protocol"])


class LogSigne(BaseEstimator, TransformerMixin):
    """x -> signe(x) · log(1 + |x|). Écrase les queues très longues (débits en octets/s
    de 0 à 10^9) avant la normalisation. Utile pour la régression logistique et le MLP."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype="float64")
        return (np.sign(X) * np.log1p(np.abs(X))).astype("float32")


def separer_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feats = caracteristiques(df)
    X = encoder_protocol(df[feats].astype("float32"))
    y = df[config.CIBLE].astype(str)
    return X, y


def decouper(df: pd.DataFrame, test: float = 0.15, val: float = 0.15,
             graine: int = config.GRAINE):
    """Découpage stratifié 70 / 15 / 15 : chaque classe garde la même proportion
    dans les trois ensembles (indispensable pour les classes rares)."""
    X, y = separer_X_y(df)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test, stratify=y, random_state=graine)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val / (1 - test), stratify=y_tmp, random_state=graine)
    return X_train, X_val, X_test, y_train, y_val, y_test


def decouper_par_jour(df: pd.DataFrame, jours_test: list[str]):
    """Variante plus exigeante : on teste sur des JOURS jamais vus à l'entraînement.
    Attention : une attaque présente seulement ces jours-là n'est pas apprise."""
    masque = df["Jour"].isin(jours_test)
    Xa, ya = separer_X_y(df.loc[~masque])
    Xt, yt = separer_X_y(df.loc[masque])
    return Xa, Xt, ya, yt


def sauvegarder_splits(X_train, X_val, X_test, y_train, y_val, y_test,
                       dossier: Path = config.DOSSIER_SPLITS) -> LabelEncoder:
    """Enregistre les 3 ensembles + l'encodeur des classes. Le binôme travaille
    ensuite TOUJOURS sur ces mêmes fichiers : les résultats restent comparables."""
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    enc = LabelEncoder().fit(pd.concat([y_train, y_val, y_test]))
    for nom, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        X.reset_index(drop=True).to_parquet(dossier / f"X_{nom}.parquet", index=False)
        pd.DataFrame({config.CIBLE: y.to_numpy()}).to_parquet(dossier / f"y_{nom}.parquet", index=False)
    joblib.dump(enc, dossier / "encodeur_classes.joblib")
    meta = {"caracteristiques": list(X_train.columns), "classes": list(enc.classes_),
            "tailles": {"train": len(X_train), "val": len(X_val), "test": len(X_test)}}
    with open(dossier / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    resume = pd.DataFrame({"train": y_train.value_counts(), "val": y_val.value_counts(),
                           "test": y_test.value_counts()}).fillna(0).astype(int)
    resume.to_csv(dossier / "repartition_splits.csv")
    print(f"[splits] train {len(X_train):,} | val {len(X_val):,} | test {len(X_test):,} "
          f"| {X_train.shape[1]} caractéristiques -> {dossier}")
    return enc


def charger_splits(dossier: Path = config.DOSSIER_SPLITS):
    """Renvoie X_train, X_val, X_test, y_train, y_val, y_test (y encodés en entiers) et l'encodeur."""
    dossier = Path(dossier)
    enc = joblib.load(dossier / "encodeur_classes.joblib")
    out = []
    for nom in ("train", "val", "test"):
        out.append(pd.read_parquet(dossier / f"X_{nom}.parquet"))
    for nom in ("train", "val", "test"):
        y = pd.read_parquet(dossier / f"y_{nom}.parquet")[config.CIBLE]
        out.append(pd.Series(enc.transform(y), name=config.CIBLE))
    return (*out, enc)


def reequilibrer(X, y, plafond: int | None = 200_000, plancher: int | None = 5_000,
                 graine: int = config.GRAINE):
    """Rééquilibrage du TRAIN uniquement.
    1) sous-échantillonnage : aucune classe au-dessus de `plafond` lignes (surtout Benign) ;
    2) SMOTE : les classes sous `plancher` lignes sont complétées par des exemples
       synthétiques, interpolés entre un exemple et l'un de ses k plus proches voisins.
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    comptes = pd.Series(y).value_counts()
    if plafond is not None:
        strat = {k: min(v, plafond) for k, v in comptes.items()}
        X, y = RandomUnderSampler(sampling_strategy=strat, random_state=graine).fit_resample(X, y)
        comptes = pd.Series(y).value_counts()
    if plancher is not None:
        cibles = {k: plancher for k, v in comptes.items() if v < plancher}
        if cibles:
            k = max(1, min(5, int(comptes[list(cibles)].min()) - 1))
            X, y = SMOTE(sampling_strategy=cibles, k_neighbors=k,
                         random_state=graine).fit_resample(X, y)
    return X, y
