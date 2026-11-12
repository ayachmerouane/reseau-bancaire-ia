"""Étape 6 : ÉVALUATION des modèles.

Métrique principale : F1 macro. Elle fait la moyenne du F1 de chaque classe, sans
pondérer par la taille : SQL Injection (87 flux) compte autant que Benign (13 M).
L'accuracy seule est trompeuse : répondre toujours « Benign » donne déjà plus de 80 %.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas de fenêtre graphique nécessaire (fonctionne aussi dans un notebook)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)

from . import config

FICHIER_RESULTATS = config.DOSSIER_RESULTATS / "resultats.csv"


def evaluer(modele, X, y, encodeur, nom: str, ensemble: str = "test",
            params: dict | None = None, auteur: str = "", enregistrer: bool = True,
            dossier_figures: Path = config.DOSSIER_FIGURES) -> dict:
    """Calcule les métriques, trace la matrice de confusion et ajoute une ligne
    dans resultats/resultats.csv (le tableau comparatif partagé du binôme)."""
    debut = time.perf_counter()
    y_pred = np.asarray(modele.predict(X)).ravel().astype(int)
    duree = time.perf_counter() - debut
    noms = list(encodeur.classes_)
    etiquettes = list(range(len(noms)))

    m = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "modele": nom, "ensemble": ensemble, "auteur": auteur,
        "f1_macro": f1_score(y, y_pred, average="macro", labels=etiquettes, zero_division=0),
        "f1_pondere": f1_score(y, y_pred, average="weighted", labels=etiquettes, zero_division=0),
        "precision_macro": precision_score(y, y_pred, average="macro", labels=etiquettes, zero_division=0),
        "rappel_macro": recall_score(y, y_pred, average="macro", labels=etiquettes, zero_division=0),
        "accuracy": accuracy_score(y, y_pred),
        "accuracy_equilibree": balanced_accuracy_score(y, y_pred),
        "ms_pour_1000_flux": 1000 * duree / max(len(X), 1) * 1000,
        "params": str(params or {}),
    }
    print(f"\n[{nom} / {ensemble}] F1 macro = {m['f1_macro']:.4f} | accuracy = {m['accuracy']:.4f} "
          f"| accuracy équilibrée = {m['accuracy_equilibree']:.4f}")

    rapport = pd.DataFrame(classification_report(
        y, y_pred, labels=etiquettes, target_names=noms, output_dict=True, zero_division=0)).T
    print(rapport.round(3).to_string())

    if enregistrer:
        FICHIER_RESULTATS.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([m]).to_csv(FICHIER_RESULTATS, mode="a", index=False,
                                 header=not FICHIER_RESULTATS.exists())
        rapport.to_csv(config.DOSSIER_RESULTATS / f"rapport_{nom}_{ensemble}.csv")
        tracer_matrice_confusion(y, y_pred, noms, nom, ensemble, dossier_figures)
    m["rapport_par_classe"] = rapport
    m["y_pred"] = y_pred
    return m


def tracer_matrice_confusion(y, y_pred, noms, nom, ensemble="test",
                             dossier: Path = config.DOSSIER_FIGURES) -> Path:
    """Matrice normalisée par ligne : chaque case = % des flux de la vraie classe
    prédits dans la classe de la colonne. La diagonale = rappel de chaque classe."""
    cm = confusion_matrix(y, y_pred, labels=range(len(noms)), normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=noms, yticklabels=noms,
                cbar_kws={"label": "proportion de la vraie classe"}, ax=ax, annot_kws={"size": 7})
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Vraie classe")
    ax.set_title(f"Matrice de confusion normalisée : {nom} ({ensemble})")
    plt.tight_layout()
    Path(dossier).mkdir(parents=True, exist_ok=True)
    chemin = Path(dossier) / f"confusion_{nom}_{ensemble}.png"
    fig.savefig(chemin, dpi=130)
    plt.close(fig)
    return chemin


def principales_confusions(y, y_pred, encodeur, n: int = 10) -> pd.DataFrame:
    """Analyse des erreurs : les couples (vraie classe -> classe prédite) les plus fréquents."""
    noms = np.array(encodeur.classes_)
    d = pd.DataFrame({"vraie": noms[np.asarray(y)], "predite": noms[np.asarray(y_pred)]})
    err = d[d.vraie != d.predite]
    if err.empty:
        return pd.DataFrame(columns=["vraie", "predite", "nb_erreurs", "pct_de_la_vraie_classe"])
    t = err.value_counts().rename("nb_erreurs").reset_index()
    total = d.vraie.value_counts()
    t["pct_de_la_vraie_classe"] = (100 * t.nb_erreurs / t.vraie.map(total)).round(2)
    return t.head(n)


def tableau_comparatif(ensemble: str = "test") -> pd.DataFrame:
    """Lit resultats.csv et garde le meilleur essai de chaque modèle."""
    if not FICHIER_RESULTATS.exists():
        return pd.DataFrame()
    r = pd.read_csv(FICHIER_RESULTATS)
    r = r[r.ensemble == ensemble].sort_values("f1_macro", ascending=False)
    return r.drop_duplicates("modele")[["modele", "f1_macro", "f1_pondere", "rappel_macro",
                                        "accuracy", "accuracy_equilibree", "ms_pour_1000_flux"]]
