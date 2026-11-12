# %% [markdown]
# # 02 : Modélisation, comparaison, optimisation et explicabilité
#
# Prérequis : `python main.py decouper` a créé `data/splits/` (le même pour tout le binôme).

# %%
import os, sys
from pathlib import Path

RACINE = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RACINE))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image

from src import config, modeles, evaluation, explicabilite
from src.pretraitement import charger_splits, reequilibrer

DONNEES = Path(os.environ.get("DONNEES", RACINE / "data"))
AUTEUR = os.environ.get("AUTEUR", "")
PLAFOND = int(os.environ.get("PLAFOND", 200_000))
PLANCHER = int(os.environ.get("PLANCHER", 5_000))
%matplotlib inline

# %%
X_train, X_val, X_test, y_train, y_val, y_test, enc = charger_splits(DONNEES / "splits")
print("train", X_train.shape, "| val", X_val.shape, "| test", X_test.shape)
pd.Series(enc.inverse_transform(y_train)).value_counts().to_frame("train (avant rééquilibrage)")

# %% [markdown]
# ## Rééquilibrage du train uniquement
# Benign plafonné, classes rares complétées par SMOTE. Le test garde la distribution réelle.

# %%
X_res, y_res = reequilibrer(X_train, y_train, plafond=PLAFOND, plancher=PLANCHER)
pd.Series(enc.inverse_transform(y_res)).value_counts().to_frame("train après rééquilibrage")

# %% [markdown]
# ## Entraînement et comparaison
# Du plus simple au plus complexe. La régression logistique est la **référence** : tout
# autre modèle doit faire mieux pour justifier sa complexité.

# %%
resultats = {}
for nom in ["regression_logistique", "arbre_decision", "random_forest", "xgboost", "lightgbm", "mlp"]:
    m, duree = modeles.entrainer(nom, X_res, y_res)
    r = evaluation.evaluer(m, X_test, y_test, enc, nom, "test", auteur=AUTEUR,
                           params={"entrainement_s": round(duree, 1)})
    modeles.sauvegarder_modele(m, nom)
    resultats[nom] = (m, r)

# %%
evaluation.tableau_comparatif()

# %% [markdown]
# ## Analyse des erreurs du meilleur modèle

# %%
meilleur = max(resultats, key=lambda k: resultats[k][1]["f1_macro"])
print("Meilleur modèle :", meilleur)
display(resultats[meilleur][1]["rapport_par_classe"].round(3))
display(evaluation.principales_confusions(y_test, resultats[meilleur][1]["y_pred"], enc))
Image(filename=str(config.DOSSIER_FIGURES / f"confusion_{meilleur}_test.png"))

# %% [markdown]
# ## Effet du rééquilibrage : même modèle, sans SMOTE ni sous-échantillonnage

# %%
m_brut, _ = modeles.entrainer("lightgbm", X_train, y_train)
_ = evaluation.evaluer(m_brut, X_test, y_test, enc, "lightgbm_sans_reequilibrage", "test", auteur=AUTEUR)

# %% [markdown]
# ## Optimisation des hyperparamètres (Optuna)
# Objectif : F1 macro sur la **validation**. Le test n'est utilisé qu'une fois, à la fin.

# %%
N_ESSAIS = int(os.environ.get("N_ESSAIS", 20))
etude = modeles.optimiser("xgboost", X_res, y_res, X_val, y_val, n_essais=N_ESSAIS)
m_opt, _ = modeles.entrainer("xgboost", X_res, y_res, etude.best_params)
_ = evaluation.evaluer(m_opt, X_test, y_test, enc, "xgboost_optuna", "test",
                       params=etude.best_params, auteur=AUTEUR)
modeles.sauvegarder_modele(m_opt, "xgboost_optuna")
etude.trials_dataframe()[["number", "value", "params_max_depth", "params_learning_rate"]].sort_values("value", ascending=False).head()

# %% [markdown]
# ## Explicabilité (SHAP) sur le modèle XGBoost optimisé

# %%
expl, X_ech = explicabilite.valeurs_shap(m_opt, X_test, n_max=1500)
imp = explicabilite.importance_shap_par_classe(expl, enc)
display(imp.head(15).round(4))
classes = [c for c in ["SSH-Bruteforce", "DoS attacks-Hulk", "Bot", "Infilteration"] if c in enc.classes_]
for c in explicabilite.tracer_shap(expl, enc, "xgboost_optuna", classes=classes):
    display(Image(filename=str(c)))
