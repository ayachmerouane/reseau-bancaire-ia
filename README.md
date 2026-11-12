# Partie 2 : classification des attaques réseau (CSE-CIC-IDS2018)

Projet « Réseau bancaire sécurisé », module Réseaux et protocoles de communication.
Objectif : à partir des statistiques d'un flux réseau, prédire s'il est normal ou de
**quel type d'attaque** il s'agit (15 classes).

## Organisation

```
projet-ia/
├── main.py                      commandes de toute la chaîne (voir plus bas)
├── app.py                       démo Streamlit pour la soutenance
├── src/
│   ├── config.py                chemins, graine, colonnes à exclure
│   ├── chargement.py            CSV -> Parquet, rechargement avec échantillonnage
│   ├── audit.py                 16 contrôles qualité AVANT nettoyage (ne modifie rien)
│   ├── nettoyage.py             nettoyage + journal des lignes retirées par classe
│   ├── pretraitement.py         one-hot Protocol, split 70/15/15, SMOTE
│   ├── modeles.py               7 modèles + optimisation Optuna
│   ├── evaluation.py            F1 macro, matrice de confusion, tableau comparatif
│   ├── explicabilite.py         SHAP par classe
│   └── deep_learning.py         MLP et CNN 1D avec Keras
├── notebooks/
│   ├── 01_exploration_nettoyage.ipynb   audit pas à pas, graphiques, nettoyage
│   └── 02_modeles.ipynb                 entraînement, comparaison, Optuna, SHAP
├── tests/
│   ├── generer_donnees_test.py  faux CSV avec les mêmes défauts que les vrais
│   └── test_pipeline.py         test de bout en bout
├── data/                        (non versionné) raw/ parquet/ clean/ splits/
├── resultats/                   audit/, figures/, resultats.csv (tableau commun)
└── modeles/                     modèles entraînés (.joblib)
```

Répartition : **IA 1** = chargement, audit, nettoyage, exploration, SHAP.
**IA 2** = prétraitement, modèles, évaluation, Optuna, Deep Learning, démo.

## 1. Installation (Windows, PowerShell)

```powershell
cd "C:\Reseau bancaire securise\projet-ia"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# seulement pour la partie Deep Learning :
pip install -r requirements-deep.txt
```

Si PowerShell refuse `Activate.ps1` : lancer une fois
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, puis réessayer.
Avec `cmd.exe` au lieu de PowerShell : `.venv\Scripts\activate.bat`.

Sous Linux, macOS ou Colab : `python3 -m venv .venv && source .venv/bin/activate`.

## 2. Vérifier que tout marche, SANS les vraies données (2 minutes)

On génère de faux CSV (mêmes colonnes et mêmes défauts : en-têtes répétés, inf,
NaN, doublons, colonnes en plus le 20/02) et on lance toute la chaîne dessus :

```powershell
python -m pytest -q tests
```

Ou étape par étape, pour voir chaque sortie :

```powershell
python tests\generer_donnees_test.py --sortie data_test\raw --lignes 3000
python main.py --donnees data_test convertir
python main.py --donnees data_test audit-complet
python main.py --donnees data_test audit
python main.py --donnees data_test nettoyer
python main.py --donnees data_test decouper
python main.py --donnees data_test entrainer --modele tous --plafond 3000 --plancher 300
python main.py --donnees data_test optimiser --modele xgboost --essais 3 --plafond 3000 --plancher 300
python main.py --donnees data_test expliquer --modele xgboost --n 300
```

Les scores sur ces fausses données sont proches de 100 % : c'est normal, elles sont
faciles. Ils ne veulent rien dire, seul le bon déroulement compte. Pensez ensuite à
vider `resultats\` et `modeles\` avant de passer aux vraies données.

## 3. Avec les vraies données

Copier les 10 CSV dans `data\raw\` (le nom des fichiers n'a pas d'importance).

```powershell
# a) Conversion en Parquet : une seule fois (10 à 20 min). Une seule personne la fait,
#    puis partage data\parquet\ (Drive ou dataset Kaggle privé).
python main.py convertir

# b) AUDIT, avant de nettoyer quoi que ce soit
python main.py audit-complet                    # comptages EXACTS sur les 16 M de lignes
python main.py audit --frac-benign 0.05         # les 16 contrôles sur un échantillon
#    -> résultats dans resultats\audit_complet\ et resultats\audit\ (un CSV par contrôle)

# c) Nettoyage (garde toutes les attaques, 10 % des Benign, au plus 300 000 par classe)
python main.py nettoyer --frac-benign 0.1 --max-par-classe 300000
#    -> data\clean\journal_nettoyage.csv et bilan_classes.csv : à mettre dans le rapport

# d) Découpage figé, le même pour tout le binôme
python main.py decouper

# e) Modèles
python main.py entrainer --modele regression_logistique --auteur IA2
python main.py entrainer --modele xgboost --auteur IA2
python main.py entrainer --modele tous --auteur IA2
python main.py optimiser --modele xgboost --essais 30
python main.py expliquer --modele xgboost_optuna --classes "SSH-Bruteforce" "Bot" "Infilteration"
python main.py deep --architecture mlp
python main.py deep --architecture cnn1d

# f) Démo
streamlit run app.py
```

Aide de chaque commande : `python main.py entrainer --help`.

Modèles disponibles : `regression_logistique`, `knn`, `arbre_decision`,
`random_forest`, `xgboost`, `lightgbm`, `mlp` (ou `tous`, sans KNN qui est trop lent).

### Notebooks

```powershell
jupyter notebook notebooks
```

`01_exploration_nettoyage` reprend l'audit contrôle par contrôle, avec un texte
explicatif et les graphiques du rapport (répartition des classes, boîtes à moustaches,
corrélations, PCA), puis le nettoyage. `02_modeles` entraîne, compare, optimise et
explique. Les notebooks sont aussi fournis en `.py` : ils sont plus lisibles dans
Git et se convertissent avec `jupytext --to ipynb notebooks\02_modeles.py`.

Réglages via variables d'environnement, par exemple :
`$env:FRAC_BENIGN="0.05"; $env:N_ESSAIS="10"; jupyter notebook notebooks`.

## 4. L'audit avant nettoyage

`python main.py audit` ne modifie rien. Il mesure chaque type de problème et enregistre
un CSV par contrôle dans `resultats\audit\` :

| # | Contrôle | Ce qu'on cherche |
|---|---|---|
| 1 | Vue d'ensemble, types | Taille, mémoire, colonnes texte ou numériques |
| 2 | Défauts de lecture | En-têtes répétés, textes dans des colonnes numériques, colonnes en plus le 20/02 |
| 3 | Valeurs manquantes | NaN par colonne |
| 4 | Valeurs infinies | `inf` dans `Flow Byts/s` et `Flow Pkts/s` (division par une durée nulle) |
| 5 | NaN ou inf par classe | Quelle classe perdrait des lignes si on les supprime |
| 6 | Doublons | 3 définitions : toutes colonnes, sans horaire ni IP, Label ignoré |
| 7 | Doublons par classe | Classes gonflées artificiellement |
| 8 | Conflits d'étiquettes | Mêmes caractéristiques, étiquettes différentes |
| 9 | Colonnes constantes | Variance nulle ou valeur dominante à plus de 99,9 % |
| 10 | Colonnes identiques | Deux colonnes strictement égales |
| 11 | Valeurs négatives | Impossibles pour une durée ou une taille (`Init * Win Byts = -1` est autorisé) |
| 12 | Valeurs aberrantes | Méthode IQR : signalées, **pas supprimées** (une attaque est souvent extrême) |
| 13 | Domaines de validité | `Protocol` dans {0, 6, 17}, `Dst Port` dans [0, 65535] |
| 14 | Répartition des classes | Déséquilibre (ratio par rapport à la classe majoritaire) |
| 15 | Corrélations fortes | Paires avec \|r\| > 0,95 |
| 16 | Types des colonnes | Colonnes à valeurs entières stockées en flottant |

Le nettoyage (`src/nettoyage.py`) applique ensuite, dans l'ordre : retrait des colonnes
de fuite (`Timestamp`, IP), inf -> NaN, suppression des NaN, des négatifs impossibles,
des doublons (**avant** le split), des conflits, des colonnes constantes et identiques.
Chaque étape est tracée dans le journal, avec les pertes par classe.



## Référence du jeu de données

CSE-CIC-IDS2018, Communications Security Establishment et Canadian Institute for
Cybersecurity, https://www.unb.ca/cic/datasets/ids-2018.html.
I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, « Toward Generating a New Intrusion
Detection Dataset and Intrusion Traffic Characterization », ICISSP 2018.
