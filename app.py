"""Démo de soutenance : la « sonde IDS » de la banque.

Lancer :  streamlit run app.py
On charge des flux (CSV au format CICFlowMeter, ou un échantillon du jeu de test),
le modèle prédit le type d'attaque, et l'application affiche la contre-mesure prévue
dans l'infrastructure réseau de la partie 1.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.pretraitement import encoder_protocol

CONTRE_MESURES = {
    "Benign": "Aucune action.",
    "SSH-Bruteforce": "Bloquer l'IP source par ACL sur le routeur de bordure, alerter le SOC.",
    "FTP-BruteForce": "Bloquer l'IP source par ACL, désactiver FTP au profit de SFTP.",
    "Brute Force -Web": "Bloquer l'IP source au pare-feu ASA, verrouiller le compte visé.",
    "Brute Force -XSS": "Filtrer la requête (WAF), corriger l'application web en DMZ.",
    "SQL Injection": "Bloquer l'IP source, isoler le serveur web en DMZ, audit de la base.",
    "Web Attack": "Bloquer l'IP source au pare-feu ASA, vérifier le serveur web en DMZ.",
    "Bot": "Isoler le poste infecté (VLAN de quarantaine), bloquer le serveur C&C.",
    "Infilteration": "Isoler le poste interne, analyse antivirale, revue des accès.",
}


def contre_mesure(classe: str) -> str:
    if classe in CONTRE_MESURES:
        return CONTRE_MESURES[classe]
    if "DoS" in classe or "DDOS" in classe or "DDoS" in classe:
        return "Limiter le débit (rate limiting) et filtrer en amont chez le FAI."
    return "Alerter l'équipe sécurité."


@st.cache_resource
def charger_modele(chemin):
    return joblib.load(chemin)


def preparer(df: pd.DataFrame, colonnes: list[str]) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if config.CIBLE in df:
        df = df[df[config.CIBLE] != config.CIBLE]
    X = df.drop(columns=[c for c in config.COLONNES_FUITE + [config.CIBLE, "Jour"] if c in df.columns])
    X = X.apply(pd.to_numeric, errors="coerce").astype("float32")
    X = encoder_protocol(X)
    X = X.replace([np.inf, -np.inf], np.nan)
    manquantes = [c for c in colonnes if c not in X.columns]
    if manquantes:
        st.warning(f"Colonnes absentes, mises à 0 : {', '.join(manquantes[:8])}…")
    return X.reindex(columns=colonnes).fillna(0)


st.set_page_config(page_title="Sonde IDS bancaire", layout="wide")
st.title("Sonde IDS du réseau bancaire : classification des flux")

dossier_splits = Path(st.sidebar.text_input("Dossier des splits", str(config.DOSSIER_SPLITS)))
dossier_modeles = Path(st.sidebar.text_input("Dossier des modèles", str(config.DOSSIER_MODELES)))
fichiers = sorted(dossier_modeles.glob("*.joblib"))
if not fichiers or not (dossier_splits / "meta.json").exists():
    st.error("Aucun modèle ou aucun split trouvé : lancez d'abord `python main.py decouper` "
             "puis `python main.py entrainer --modele xgboost`.")
    st.stop()

nom = st.sidebar.selectbox("Modèle", [f.stem for f in fichiers])
modele = charger_modele(dossier_modeles / f"{nom}.joblib")
meta = json.loads((dossier_splits / "meta.json").read_text(encoding="utf-8"))
classes = np.array(meta["classes"])

source = st.radio("Source des flux", ["Échantillon du jeu de test", "Fichier CSV (format CICFlowMeter)"],
                  horizontal=True)
vrai = None
if source.startswith("Échantillon"):
    n = st.slider("Nombre de flux", 50, 5000, 500, step=50)
    X = pd.read_parquet(dossier_splits / "X_test.parquet").sample(n, random_state=0)
    vrai = pd.read_parquet(dossier_splits / "y_test.parquet").loc[X.index, config.CIBLE].to_numpy()
else:
    f = st.file_uploader("CSV de flux", type="csv")
    if f is None:
        st.stop()
    brut = pd.read_csv(f, low_memory=False)
    if config.CIBLE in brut:
        vrai = brut.loc[brut[config.CIBLE] != config.CIBLE, config.CIBLE].astype(str).to_numpy()
    X = preparer(brut, meta["caracteristiques"])

pred = classes[np.asarray(modele.predict(X)).ravel().astype(int)]
res = pd.DataFrame({"prediction": pred})
if vrai is not None:
    res.insert(0, "vraie_classe", vrai)
res["contre_mesure"] = res["prediction"].map(contre_mesure)

c1, c2, c3 = st.columns(3)
c1.metric("Flux analysés", len(res))
c2.metric("Flux suspects", int((res.prediction != "Benign").sum()))
if vrai is not None:
    c3.metric("Prédictions correctes", f"{(res.prediction == res.vraie_classe).mean():.1%}")

st.subheader("Alertes par type d'attaque")
alertes = res[res.prediction != "Benign"].prediction.value_counts()
st.bar_chart(alertes)
st.subheader("Contre-mesures déclenchées")
st.table(pd.DataFrame({"flux": alertes, "contre-mesure": [contre_mesure(c) for c in alertes.index]}))
st.subheader("Détail des flux")
st.dataframe(res, use_container_width=True)
