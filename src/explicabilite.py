"""Étape 7 : EXPLICABILITÉ avec SHAP et importance des caractéristiques.

SHAP attribue à chaque caractéristique sa contribution à une prédiction. Pour un IDS,
cela répond à la question de l'analyste : « pourquoi ce flux a-t-il été classé SSH-Bruteforce ? »
TreeExplainer est exact et rapide pour les modèles à arbres (RF, XGBoost, LightGBM).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def importance_native(modele, colonnes, n: int = 20) -> pd.DataFrame:
    """Importance intégrée au modèle (gain moyen des arbres)."""
    est = modele.steps[-1][1] if hasattr(modele, "steps") else modele
    if not hasattr(est, "feature_importances_"):
        raise ValueError("Ce modèle n'a pas d'importance native (utiliser SHAP ou la permutation).")
    t = pd.Series(est.feature_importances_, index=colonnes, name="importance")
    return t.sort_values(ascending=False).head(n).to_frame()


def valeurs_shap(modele, X: pd.DataFrame, n_max: int = 2000):
    """Renvoie (explication shap, échantillon utilisé). Tableau de forme
    (lignes, caractéristiques, classes) pour un classifieur multi-classes."""
    import shap
    Xe = X.sample(min(n_max, len(X)), random_state=config.GRAINE)
    explainer = shap.TreeExplainer(modele)
    return explainer(Xe), Xe


def importance_shap_par_classe(expl, encodeur) -> pd.DataFrame:
    """Moyenne des |SHAP| : lignes = caractéristiques, colonnes = classes."""
    v = np.abs(expl.values)
    if v.ndim == 2:  # cas binaire
        v = v[:, :, None]
    t = pd.DataFrame(v.mean(axis=0), index=expl.feature_names,
                     columns=list(encodeur.classes_)[: v.shape[2]])
    t["global"] = t.mean(axis=1)
    return t.sort_values("global", ascending=False)


def tracer_shap(expl, encodeur, nom: str, classes: list[str] | None = None,
                dossier: Path = config.DOSSIER_FIGURES, n: int = 15) -> list[Path]:
    """Une figure globale + un « beeswarm » par classe demandée."""
    import shap
    Path(dossier).mkdir(parents=True, exist_ok=True)
    chemins = []
    imp = importance_shap_par_classe(expl, encodeur)
    fig, ax = plt.subplots(figsize=(9, 6))
    imp["global"].head(n)[::-1].plot.barh(ax=ax, color="#2b6cb0")
    ax.set_xlabel("moyenne |valeur SHAP| (toutes classes)")
    ax.set_title(f"Caractéristiques les plus influentes : {nom}")
    plt.tight_layout()
    c = Path(dossier) / f"shap_global_{nom}.png"
    fig.savefig(c, dpi=130)
    plt.close(fig)
    chemins.append(c)

    noms = list(encodeur.classes_)
    for classe in classes or []:
        k = noms.index(classe)
        plt.figure()
        shap.plots.beeswarm(expl[:, :, k], max_display=n, show=False)
        plt.title(f"SHAP : {classe}")
        plt.tight_layout()
        c = Path(dossier) / f"shap_{nom}_{classe.replace(' ', '_').replace('/', '-')}.png"
        plt.savefig(c, dpi=130)
        plt.close("all")
        chemins.append(c)
    return chemins
