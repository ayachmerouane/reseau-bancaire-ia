"""Test de bout en bout sur de fausses données (quelques secondes).

Lancer :  python -m pytest -q tests
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent


def lancer(*args, cwd=RACINE):
    r = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return r.stdout


def test_chaine_complete(tmp_path):
    donnees, res, mod = tmp_path / "data", tmp_path / "res", tmp_path / "mod"
    lancer("tests/generer_donnees_test.py", "--sortie", str(donnees / "raw"), "--lignes", "800")
    O = ["main.py", "--donnees", str(donnees), "--resultats", str(res), "--modeles", str(mod)]

    lancer(*O, "convertir")
    assert len(list((donnees / "parquet").glob("*.parquet"))) == 10

    sortie = lancer(*O, "audit")
    assert "Doublons" in sortie and (res / "audit" / "06_doublons.csv").exists()

    lancer(*O, "nettoyer")
    propre = pd.read_parquet(donnees / "clean" / "donnees_propres.parquet")
    assert "Timestamp" not in propre and "Src IP" not in propre
    assert not propre.drop(columns=["Label", "Jour"]).isna().any().any()
    assert not propre.duplicated().any()

    lancer(*O, "decouper")
    lancer(*O, "entrainer", "--modele", "random_forest", "--plafond", "1000", "--plancher", "100")
    r = pd.read_csv(res / "resultats.csv")
    assert (r.modele == "random_forest").any() and r.f1_macro.max() > 0.5
