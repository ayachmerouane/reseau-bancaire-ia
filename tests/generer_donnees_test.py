"""Génère de faux CSV au format CSE-CIC-IDS2018 pour tester le code sans les 7 Go.

Les fichiers reproduisent volontairement les défauts du vrai jeu de données :
lignes d'en-tête répétées, valeurs inf et NaN, doublons, colonne constante,
4 colonnes en plus dans le fichier du 20/02, valeurs -1 dans Init Fwd Win Byts.

Usage :
    python tests/generer_donnees_test.py --sortie data/raw_test --lignes 3000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLONNES = [
    "Dst Port", "Protocol", "Timestamp", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Tot", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Tot", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s",
    "Bwd Pkts/s", "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std",
    "Pkt Len Var", "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
    "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt", "Down/Up Ratio",
    "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg", "Fwd Byts/b Avg",
    "Fwd Pkts/b Avg", "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg",
    "Bwd Blk Rate Avg", "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts",
    "Subflow Bwd Byts", "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts",
    "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label",
]

# jour -> attaques présentes ce jour-là (comme dans le vrai jeu)
JOURS = {
    "02-14-2018": ["FTP-BruteForce", "SSH-Bruteforce"],
    "02-15-2018": ["DoS attacks-GoldenEye", "DoS attacks-Slowloris"],
    "02-16-2018": ["DoS attacks-SlowHTTPTest", "DoS attacks-Hulk"],
    "02-20-2018": ["DDoS attacks-LOIC-HTTP"],
    "02-21-2018": ["DDOS attack-LOIC-UDP", "DDOS attack-HOIC"],
    "02-22-2018": ["Brute Force -Web", "Brute Force -XSS", "SQL Injection"],
    "02-23-2018": ["Brute Force -Web", "Brute Force -XSS", "SQL Injection"],
    "02-28-2018": ["Infilteration"],
    "03-01-2018": ["Infilteration"],
    "03-02-2018": ["Bot"],
}
PORTS = {"FTP-BruteForce": 21, "SSH-Bruteforce": 22, "DDOS attack-LOIC-UDP": 53, "Bot": 8080}
RARES = {"Brute Force -Web": 0.03, "Brute Force -XSS": 0.015, "SQL Injection": 0.008}


def generer_jour(jour, attaques, n, rng):
    noms_num = [c for c in COLONNES if c not in ("Dst Port", "Protocol", "Timestamp", "Label")]
    # proportions : ~70 % Benign, le reste réparti entre les attaques (classes web rares)
    probas = [0.7] + [RARES.get(a, 0.3 / len(attaques)) for a in attaques]
    probas = np.array(probas) / sum(probas)
    labels = rng.choice(["Benign"] + attaques, size=n, p=probas)

    df = pd.DataFrame(index=range(n))
    for i, lab in enumerate(["Benign"] + attaques):
        m = labels == lab
        k = m.sum()
        # chaque classe a sa propre « signature » statistique
        centre = rng.uniform(1, 50, size=len(noms_num)) * (1 + i)
        bloc = np.abs(rng.normal(centre, centre * 0.3, size=(k, len(noms_num))))
        if lab == "Infilteration":  # ressemble volontairement à du Benign
            bloc = np.abs(rng.normal(25, 8, size=(k, len(noms_num))))
        df.loc[m, noms_num] = bloc
        df.loc[m, "Dst Port"] = PORTS.get(lab, rng.choice([80, 443, 3389, 445]))
        df.loc[m, "Protocol"] = 17 if lab == "DDOS attack-LOIC-UDP" else 6
    df.loc[rng.random(n) < 0.05, "Protocol"] = 0
    df["Dst Port"] = df["Dst Port"].astype(int)
    df["Protocol"] = df["Protocol"].astype(int)
    df["Bwd PSH Flags"] = 0  # colonne constante, comme dans le vrai jeu
    df.loc[rng.random(n) < 0.1, "Init Fwd Win Byts"] = -1
    debut = pd.Timestamp(f"2018-{jour[:2]}-{jour[3:5]} 08:00:00")
    df["Timestamp"] = (debut + pd.to_timedelta(rng.integers(0, 36000, n), unit="s")).strftime("%d/%m/%Y %H:%M:%S")
    df["Label"] = labels
    df = df[COLONNES]

    # défauts : inf et NaN sur les débits, doublons, flux négatif
    idx = rng.choice(n, size=max(1, n // 200), replace=False)
    df.loc[idx, "Flow Byts/s"] = np.inf
    df.loc[idx[: len(idx) // 2], "Flow Pkts/s"] = np.nan
    df.loc[rng.choice(n, size=3, replace=False), "Flow Duration"] = -5
    df = pd.concat([df, df.sample(frac=0.05, random_state=1)], ignore_index=True)

    if jour == "02-20-2018":  # 4 colonnes en plus
        df.insert(0, "Dst IP", "172.31.69.25")
        df.insert(0, "Src Port", rng.integers(1024, 65535, len(df)))
        df.insert(0, "Src IP", "18.218.115.60")
        df.insert(0, "Flow ID", "flow")
    return df


def ecrire_avec_entetes_repetes(df, chemin, rng):
    """Écrit le CSV en insérant quelques lignes d'en-tête au milieu (défaut réel)."""
    bornes = np.linspace(0, len(df), 5).astype(int)
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        for debut, fin in zip(bornes[:-1], bornes[1:]):
            df.iloc[debut:fin].to_csv(f, index=False, header=True)  # l'en-tête est réécrit à chaque morceau


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sortie", default="data/raw_test")
    p.add_argument("--lignes", type=int, default=3000, help="lignes par fichier")
    a = p.parse_args()
    rng = np.random.default_rng(0)
    sortie = Path(a.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    for jour, attaques in JOURS.items():
        df = generer_jour(jour, attaques, a.lignes, rng)
        ecrire_avec_entetes_repetes(df, sortie / f"{jour}.csv", rng)
        print(f"{jour}.csv : {len(df)} lignes, {df.shape[1]} colonnes")


if __name__ == "__main__":
    main()
