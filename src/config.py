"""Paramètres communs à tout le projet.

Tous les chemins sont relatifs à la racine du projet et construits avec pathlib,
ce qui fonctionne aussi bien sous Windows que sous Linux ou macOS.
"""
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent

DOSSIER_CSV = RACINE / "data" / "raw"          # les 10 CSV téléchargés
DOSSIER_PARQUET = RACINE / "data" / "parquet"  # CSV convertis (un fichier par jour)
DOSSIER_PROPRE = RACINE / "data" / "clean"     # données nettoyées
DOSSIER_SPLITS = RACINE / "data" / "splits"    # train / val / test figés
DOSSIER_RESULTATS = RACINE / "resultats"
DOSSIER_FIGURES = DOSSIER_RESULTATS / "figures"
DOSSIER_MODELES = RACINE / "modeles"

# Graine aléatoire unique, partagée par tout le binôme
GRAINE = 42

CIBLE = "Label"

# Colonnes présentes seulement dans le fichier du 20/02 (identifiants de connexion)
COLONNES_IDENTIFIANTS = ["Flow ID", "Src IP", "Src Port", "Dst IP"]

# Colonnes à exclure de l'apprentissage pour éviter la fuite d'information :
# le modèle doit apprendre le comportement d'un flux, pas l'heure ni la machine.
COLONNES_FUITE = ["Timestamp"] + COLONNES_IDENTIFIANTS

# Colonnes catégorielles (codes, pas des quantités)
COLONNES_CATEGORIELLES = ["Protocol"]

# Colonnes où une valeur négative n'a pas de sens physique (durées, tailles, débits).
# Exception connue : Init Fwd Win Byts / Init Bwd Win Byts valent -1 quand la fenêtre
# TCP n'a pas été observée ; c'est un code, pas une erreur.
COLONNES_NEG_AUTORISE = ["Init Fwd Win Byts", "Init Bwd Win Byts"]

# Attaques web très rares, regroupables en une seule classe si besoin
CLASSES_WEB = ["Brute Force -Web", "Brute Force -XSS", "SQL Injection"]

for d in (DOSSIER_PARQUET, DOSSIER_PROPRE, DOSSIER_SPLITS, DOSSIER_FIGURES, DOSSIER_MODELES):
    d.mkdir(parents=True, exist_ok=True)
