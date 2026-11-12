"""Étape 5 : MODÈLES. Du plus simple (référence) au plus avancé.

Chaque modèle est une fonction qui renvoie un estimateur scikit-learn non entraîné.
Les modèles sensibles à l'échelle (régression logistique, KNN, MLP) sont des Pipeline
qui incluent leur normalisation : elle est donc ajustée sur le train uniquement.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from . import config
from .pretraitement import LogSigne

G = config.GRAINE


def regression_logistique(**p):
    return make_pipeline(LogSigne(), StandardScaler(),
                         LogisticRegression(max_iter=p.get("max_iter", 1000), C=p.get("C", 1.0),
                                            class_weight="balanced", random_state=G))


def knn(**p):
    # lent en prédiction sur des millions de lignes : à tester sur un échantillon
    return make_pipeline(LogSigne(), StandardScaler(),
                         KNeighborsClassifier(n_neighbors=p.get("n_neighbors", 5), n_jobs=-1))


def arbre_decision(**p):
    return DecisionTreeClassifier(max_depth=p.get("max_depth", 25), class_weight="balanced",
                                  random_state=G)


def random_forest(**p):
    return RandomForestClassifier(n_estimators=p.get("n_estimators", 200),
                                  max_depth=p.get("max_depth", None),
                                  min_samples_leaf=p.get("min_samples_leaf", 1),
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=G)


def xgboost(**p):
    return XGBClassifier(n_estimators=p.get("n_estimators", 400), max_depth=p.get("max_depth", 8),
                         learning_rate=p.get("learning_rate", 0.1),
                         subsample=p.get("subsample", 0.9),
                         colsample_bytree=p.get("colsample_bytree", 0.9),
                         tree_method="hist", n_jobs=-1, random_state=G, eval_metric="mlogloss")


def lightgbm(**p):
    return LGBMClassifier(n_estimators=p.get("n_estimators", 400),
                          num_leaves=p.get("num_leaves", 63),
                          learning_rate=p.get("learning_rate", 0.1),
                          class_weight="balanced", n_jobs=-1, random_state=G, verbose=-1)


def mlp(**p):
    return make_pipeline(LogSigne(), StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=p.get("couches", (128, 64)),
                                       alpha=p.get("alpha", 1e-4), batch_size=512,
                                       max_iter=p.get("max_iter", 50), early_stopping=True,
                                       random_state=G))


CATALOGUE = {
    "regression_logistique": regression_logistique,
    "knn": knn,
    "arbre_decision": arbre_decision,
    "random_forest": random_forest,
    "xgboost": xgboost,
    "lightgbm": lightgbm,
    "mlp": mlp,
}

# XGBoost n'a pas de class_weight : on lui passe des poids par ligne
POIDS_PAR_LIGNE = {"xgboost"}


def entrainer(nom: str, X, y, params: dict | None = None):
    params = params or {}
    modele = CATALOGUE[nom](**params)
    debut = time.perf_counter()
    if nom in POIDS_PAR_LIGNE:
        modele.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        modele.fit(X, y)
    duree = time.perf_counter() - debut
    print(f"[entraînement] {nom}: {duree:.1f} s sur {len(X):,} lignes")
    return modele, duree


def sauvegarder_modele(modele, nom: str, dossier: Path = config.DOSSIER_MODELES) -> Path:
    Path(dossier).mkdir(parents=True, exist_ok=True)
    chemin = Path(dossier) / f"{nom}.joblib"
    joblib.dump(modele, chemin)
    return chemin


def charger_modele(nom: str, dossier: Path = config.DOSSIER_MODELES):
    return joblib.load(Path(dossier) / f"{nom}.joblib")


# ---------------------------------------------------------------- optimisation Optuna
ESPACES = {
    "xgboost": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 100, 800, step=100),
        "max_depth": t.suggest_int("max_depth", 4, 12),
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0)},
    "lightgbm": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 100, 800, step=100),
        "num_leaves": t.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.3, log=True)},
    "random_forest": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 100, 500, step=100),
        "max_depth": t.suggest_int("max_depth", 10, 40),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 5)},
}


def optimiser(nom: str, X_train, y_train, X_val, y_val, n_essais: int = 30,
              delai_s: int | None = None):
    """Cherche les hyperparamètres qui maximisent le F1 macro sur la VALIDATION
    (jamais sur le test, qui reste réservé à l'évaluation finale)."""
    import optuna
    from sklearn.metrics import f1_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objectif(t):
        params = ESPACES[nom](t)
        modele, _ = entrainer(nom, X_train, y_train, params)
        return f1_score(y_val, np.asarray(modele.predict(X_val)).ravel(), average="macro")

    etude = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=G))
    etude.optimize(objectif, n_trials=n_essais, timeout=delai_s)
    print(f"[optuna] {nom}: meilleur F1 macro val = {etude.best_value:.4f} avec {etude.best_params}")
    return etude
