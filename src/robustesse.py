"""Expériences de robustesse : le modèle apprend-il l'attaque, ou l'environnement du labo ?

Trois questions, trois expériences (même XGBoost, mêmes paramètres, mêmes métriques) :

  E2/E3  Ablation      : complet / sans Init Fwd Win Byts / sans les deux fenêtres /
                         fenêtre binarisée (observée ou non). -> python main.py entrainer --exclure ...
  E4     Jours exclus  : on teste sur une JOURNÉE de capture jamais vue à l'entraînement
                         (validation groupée par jour, au lieu d'un split aléatoire).
  E5     SHAP comparé  : quelles caractéristiques prennent le relais quand on retire la fenêtre.

ATTENTION, propre à CSE-CIC-IDS2018 : presque chaque attaque n'existe que sur UN seul jour.
Exclure un jour retire donc souvent toute une classe de l'entraînement. D'où deux modes :
  - binaire (normal / attaque), tous les jours tour à tour exclus : mesure la détection
    d'attaques JAMAIS VUES (proche d'une attaque « zero-day ») ;
  - multi-classe, seulement sur les jours dont toutes les classes existent aussi un autre
    jour (attaques web : 22 et 23/02 ; infiltration : 28/02 et 01/03).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from . import config
from .pretraitement import separer_X_y

FENETRES = ["Init Fwd Win Byts", "Init Bwd Win Byts"]


def _binariser(X):
    X = X.copy()
    for c in FENETRES:
        if c in X:
            X[c + " observee"] = (X[c] >= 0).astype("float32")
            X = X.drop(columns=[c])
    return X


VARIANTES = {
    "complet": lambda X: X,
    "sans_fwd_win": lambda X: X.drop(columns=["Init Fwd Win Byts"], errors="ignore"),
    "sans_fenetres": lambda X: X.drop(columns=FENETRES, errors="ignore"),
    "fenetre_binaire": _binariser,
    "sans_fenetres_ni_port": lambda X: X.drop(columns=FENETRES + ["Dst Port"], errors="ignore"),
}


def donnees_avec_jour(df: pd.DataFrame):
    X, y = separer_X_y(df)
    return X.reset_index(drop=True), y.reset_index(drop=True), df["Jour"].astype(str).reset_index(drop=True)


def _plafonner(idx, y, plafond, rng):
    """Au plus `plafond` lignes par classe d'origine (garde la diversité des attaques)."""
    if plafond is None:
        return idx
    garder = []
    for _, g in pd.Series(idx).groupby(y.iloc[idx].to_numpy()):
        garder.append(g.to_numpy() if len(g) <= plafond else rng.choice(g.to_numpy(), plafond, replace=False))
    return np.concatenate(garder)


def _xgb(n_arbres, binaire):
    return XGBClassifier(n_estimators=n_arbres, max_depth=8, learning_rate=0.1, subsample=0.9,
                         colsample_bytree=0.9, tree_method="hist", n_jobs=-1,
                         random_state=config.GRAINE,
                         eval_metric="logloss" if binaire else "mlogloss")


def jours_exclus_binaire(X, y, jours, variantes, plafond=50_000, n_arbres=200) -> pd.DataFrame:
    """Chaque jour est exclu à tour de rôle (leave-one-day-out), cible binaire."""
    rng = np.random.default_rng(config.GRAINE)
    yb = (y != "Benign").astype(int)
    lignes = []
    for jour in sorted(jours.unique()):
        test = np.where(jours == jour)[0]
        train = _plafonner(np.where(jours != jour)[0], y, plafond, rng)
        attaques_test = sorted(set(y.iloc[test]) - {"Benign"})
        jamais_vues = sorted(set(attaques_test) - set(y.iloc[train]))
        for nom in variantes:
            f = VARIANTES[nom]
            Xtr, Xte = f(X.iloc[train]), f(X.iloc[test])
            debut = time.perf_counter()
            m = _xgb(n_arbres, True).fit(Xtr, yb.iloc[train],
                                         sample_weight=compute_sample_weight("balanced", yb.iloc[train]))
            proba = m.predict_proba(Xte)[:, 1]
            pred = (proba >= 0.5).astype(int)
            yt = yb.iloc[test].to_numpy()
            benign = yt == 0
            lignes.append({
                "jour_exclu": jour, "variante": nom,
                "attaques_du_jour": ", ".join(attaques_test),
                "jamais_vues_a_l_entrainement": ", ".join(jamais_vues) or "aucune",
                "rappel_attaques": recall_score(yt, pred, zero_division=0),
                "taux_fausses_alertes": float(pred[benign].mean()) if benign.any() else np.nan,
                "f1": f1_score(yt, pred, zero_division=0),
                "roc_auc": roc_auc_score(yt, proba) if len(set(yt)) == 2 else np.nan,
                "entrainement_s": round(time.perf_counter() - debut, 1),
            })
            print(f"[jours exclus, binaire] {jour[:30]:<30} {nom:<22} rappel={lignes[-1]['rappel_attaques']:.3f} "
                  f"fausses alertes={lignes[-1]['taux_fausses_alertes']:.3f}")
    return pd.DataFrame(lignes)


def jours_exclus_multiclasse(X, y, jours, variantes, plafond=50_000, n_arbres=200) -> pd.DataFrame:
    """Exclut seulement les jours dont toutes les attaques existent aussi un autre jour."""
    rng = np.random.default_rng(config.GRAINE)
    lignes, rappels = [], []
    for jour in sorted(jours.unique()):
        masque = (jours == jour).to_numpy()
        classes_jour = set(y[masque])
        classes_autres = set(y[~masque])
        if not (classes_jour - {"Benign"}) or not classes_jour <= classes_autres:
            continue  # une attaque de ce jour n'existe nulle part ailleurs
        test = np.where(masque)[0]
        train = _plafonner(np.where(~masque)[0], y, plafond, rng)
        classes = sorted(set(y.iloc[train]))
        code = {c: i for i, c in enumerate(classes)}
        ytr = y.iloc[train].map(code).to_numpy()
        yte = y.iloc[test].map(code).to_numpy()
        presentes = sorted(set(yte))
        for nom in variantes:
            f = VARIANTES[nom]
            m = _xgb(n_arbres, False).fit(f(X.iloc[train]), ytr,
                                          sample_weight=compute_sample_weight("balanced", ytr))
            pred = m.predict(f(X.iloc[test]))
            r = recall_score(yte, pred, labels=presentes, average=None, zero_division=0)
            lignes.append({"jour_exclu": jour, "variante": nom,
                           "f1_macro_classes_du_jour": f1_score(yte, pred, labels=presentes,
                                                                 average="macro", zero_division=0),
                           **{f"rappel {classes[k]}": v for k, v in zip(presentes, r)}})
            print(f"[jours exclus, multi-classe] {jour[:30]:<30} {nom:<22} "
                  f"F1 macro={lignes[-1]['f1_macro_classes_du_jour']:.3f}")
    if not lignes:
        print("Aucun jour exclu possible : chaque attaque n'existe que sur un seul jour.")
    return pd.DataFrame(lignes)


def resume(t: pd.DataFrame, colonnes) -> pd.DataFrame:
    """Moyenne ± écart type par variante, sur tous les jours exclus."""
    g = t.groupby("variante")[colonnes].agg(["mean", "std"]).round(4)
    g.columns = [f"{c}_{s}" for c, s in g.columns]
    return g


def comparer_shap(modeles: dict, X_test: pd.DataFrame, n_max=2000, top=10) -> pd.DataFrame:
    """Top des caractéristiques (moyenne |SHAP|, toutes classes) pour chaque modèle,
    calculé sur LES MÊMES flux de test."""
    import shap
    ech = X_test.sample(min(n_max, len(X_test)), random_state=config.GRAINE)
    colonnes = {}
    for nom, m in modeles.items():
        attendues = list(getattr(m, "feature_names_in_", ech.columns))
        Xm = _binariser(ech) if any(c.endswith(" observee") for c in attendues) else ech
        v = np.abs(shap.TreeExplainer(m)(Xm[attendues]).values)
        imp = v.mean(axis=(0, 2)) if v.ndim == 3 else v.mean(axis=0)
        s = pd.Series(imp, index=attendues).sort_values(ascending=False).head(top)
        colonnes[nom] = [f"{k} ({x:.3f})" for k, x in s.items()]
    return pd.DataFrame(colonnes, index=range(1, top + 1))
