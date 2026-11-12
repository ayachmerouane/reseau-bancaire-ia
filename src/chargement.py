"""Étape 1 : lire les CSV bruts et les convertir en Parquet, puis recharger les données.

Pourquoi Parquet ? Le format est colonnaire et compressé : environ 3 à 5 fois plus
léger que le CSV et beaucoup plus rapide à relire. La conversion se fait une seule fois.

Pendant la conversion, on ne « nettoie » rien, sauf ce qui empêche de lire le fichier :
les lignes d'en-tête répétées au milieu des CSV. Elles sont comptées dans un journal
(data/parquet/journal_conversion.json) pour que l'audit puisse les signaler.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config

COLONNES_TEXTE = ["Timestamp", "Label", "Flow ID", "Src IP", "Dst IP"]


def lister_csv(dossier: Path) -> list[Path]:
    fichiers = sorted(Path(dossier).glob("*.csv"))
    if not fichiers:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {dossier}")
    return fichiers


def nom_jour(chemin: Path) -> str:
    """Nom court du fichier, utilisé comme identifiant du jour de capture."""
    return Path(chemin).stem


def _convertir_morceau(morceau: pd.DataFrame, journal_fichier: dict) -> pd.DataFrame:
    morceau.columns = [c.strip() for c in morceau.columns]

    # Lignes d'en-tête répétées : la colonne Label contient le texte "Label"
    entetes = morceau[config.CIBLE] == config.CIBLE
    journal_fichier["entetes_repetes"] += int(entetes.sum())
    morceau = morceau.loc[~entetes].copy()

    for col in morceau.columns:
        if col in COLONNES_TEXTE:
            morceau[col] = morceau[col].astype("string")
            continue
        brut = morceau[col]
        num = pd.to_numeric(brut, errors="coerce")
        # valeurs texte qui n'étaient pas des nombres (hors cases vides) : on les compte
        non_num = int((num.isna() & brut.notna() & (brut.str.strip() != "")).sum())
        if non_num:
            journal_fichier["valeurs_non_numeriques"][col] = (
                journal_fichier["valeurs_non_numeriques"].get(col, 0) + non_num
            )
        morceau[col] = num.astype("float32")
    morceau["Jour"] = journal_fichier["jour"]
    morceau["Jour"] = morceau["Jour"].astype("string")
    return morceau


def convertir_csv_en_parquet(dossier_csv: Path = config.DOSSIER_CSV,
                             dossier_parquet: Path = config.DOSSIER_PARQUET,
                             taille_morceau: int = 500_000) -> dict:
    """Convertit chaque CSV en un fichier Parquet, morceau par morceau (RAM limitée)."""
    dossier_parquet = Path(dossier_parquet)
    dossier_parquet.mkdir(parents=True, exist_ok=True)
    journal = {}
    for chemin in lister_csv(dossier_csv):
        jour = nom_jour(chemin)
        sortie = dossier_parquet / f"{jour}.parquet"
        jf = {"jour": jour, "fichier": chemin.name, "lignes_lues": 0, "entetes_repetes": 0,
              "valeurs_non_numeriques": {}, "colonnes": None}
        ecrivain = None
        schema = None
        for morceau in pd.read_csv(chemin, dtype=str, chunksize=taille_morceau,
                                   keep_default_na=False, na_values=[""], encoding="utf-8"):
            jf["lignes_lues"] += len(morceau)
            morceau = _convertir_morceau(morceau, jf)
            table = pa.Table.from_pandas(morceau, preserve_index=False)
            if ecrivain is None:
                schema = table.schema
                jf["colonnes"] = list(morceau.columns)
                ecrivain = pq.ParquetWriter(sortie, schema, compression="snappy")
            ecrivain.write_table(table.cast(schema))
        if ecrivain is not None:
            ecrivain.close()
        jf["lignes_gardees"] = jf["lignes_lues"] - jf["entetes_repetes"]
        journal[jour] = jf
        print(f"[conversion] {chemin.name}: {jf['lignes_lues']:,} lignes lues, "
              f"{jf['entetes_repetes']} en-têtes répétés retirés -> {sortie.name}")
    with open(dossier_parquet / "journal_conversion.json", "w", encoding="utf-8") as f:
        json.dump(journal, f, indent=2, ensure_ascii=False)
    return journal


def lire_journal(dossier_parquet: Path = config.DOSSIER_PARQUET) -> dict:
    chemin = Path(dossier_parquet) / "journal_conversion.json"
    if not chemin.exists():
        return {}
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def lister_parquet(dossier_parquet: Path = config.DOSSIER_PARQUET) -> list[Path]:
    fichiers = sorted(Path(dossier_parquet).glob("*.parquet"))
    if not fichiers:
        raise FileNotFoundError(f"Aucun Parquet dans {dossier_parquet} : lancez d'abord "
                                "`python main.py convertir`")
    return fichiers


def charger(dossier_parquet: Path = config.DOSSIER_PARQUET,
            frac_benign: float | None = None,
            max_par_classe: int | None = None,
            garder_identifiants: bool = True,
            graine: int = config.GRAINE) -> pd.DataFrame:
    """Recharge les Parquet en un seul DataFrame, avec échantillonnage optionnel.

    frac_benign    : fraction des flux Benign gardés (ex. 0.1). Les attaques sont toutes
                     gardées à ce stade. None = tout garder.
    max_par_classe : plafond de lignes par classe après lecture (ex. 300_000).
    Les colonnes absentes d'un fichier (Src IP… hors 20/02) valent NaN.
    """
    rng = np.random.default_rng(graine)
    morceaux = []
    for chemin in lister_parquet(dossier_parquet):
        pf = pq.ParquetFile(chemin)
        for lot in pf.iter_batches(batch_size=500_000):
            df = lot.to_pandas()
            if not garder_identifiants:
                df = df.drop(columns=[c for c in config.COLONNES_IDENTIFIANTS if c in df.columns])
            if frac_benign is not None:
                benign = df[config.CIBLE] == "Benign"
                garder = ~benign | (rng.random(len(df)) < frac_benign)
                df = df.loc[garder]
            morceaux.append(df)
    df = pd.concat(morceaux, ignore_index=True)
    if max_par_classe is not None:
        # mélange puis on garde au plus max_par_classe lignes de chaque classe
        df = df.sample(frac=1.0, random_state=graine)
        df = df.groupby(config.CIBLE, observed=True).head(max_par_classe).reset_index(drop=True)
    return df


def _colonnes_hash(df: pd.DataFrame) -> list[str]:
    """Caractéristiques + Label (sans horaire, IP ni jour) : la définition des doublons."""
    exclues = set(config.COLONNES_FUITE) | {"Jour"}
    return [c for c in df.columns if c not in exclues]


def charger_pour_nettoyage(dossier_parquet: Path = config.DOSSIER_PARQUET,
                           frac_benign: float | None = None,
                           max_par_classe: int | None = None,
                           dedoublonner: bool = True,
                           graine: int = config.GRAINE,
                           taille_lot: int = 250_000) -> pd.DataFrame:
    """Chargement économe en mémoire, pour le nettoyage sur un PC portable.

    Deux passes sur les Parquet, un lot à la fois :
      1) repère les doublons (empreinte 64 bits de chaque ligne, comparée à toutes les
         lignes déjà vues) et compte les lignes UNIQUES de chaque classe ;
      2) ne garde que les lignes uniques, tirées au hasard avec une probabilité par classe
         calculée pour viser `max_par_classe` lignes (et `frac_benign` pour Benign).
    Timestamp et les IP sont retirés dès la lecture (ils seraient supprimés au nettoyage).
    On dédoublonne AVANT d'échantillonner : on garde ainsi le maximum de flux distincts.
    Le nombre de doublons retirés est noté dans df.attrs pour le journal de nettoyage.
    """
    fichiers = lister_parquet(dossier_parquet)
    vus = np.empty(0, dtype=np.uint64)
    masques, uniques, bruts = [], {}, {}

    # ---- passe 1 : doublons et comptage des lignes uniques par classe
    for chemin in fichiers:
        for lot in pq.ParquetFile(chemin).iter_batches(batch_size=taille_lot):
            df = lot.to_pandas()
            df = df.drop(columns=[c for c in config.COLONNES_FUITE if c in df.columns])
            lab = df[config.CIBLE].astype(str).to_numpy()
            if dedoublonner:
                h = pd.util.hash_pandas_object(df[_colonnes_hash(df)], index=False).to_numpy()
                _, premiers = np.unique(h, return_index=True)
                nouveau = np.zeros(len(h), dtype=bool)
                nouveau[premiers] = True                      # 1re occurrence dans le lot
                nouveau &= ~np.isin(h, vus, assume_unique=False)  # pas vu dans un lot précédent
                vus = np.union1d(vus, h[nouveau])
            else:
                nouveau = np.ones(len(df), dtype=bool)
            masques.append(nouveau)
            for k, v in pd.Series(lab).value_counts().items():
                bruts[k] = bruts.get(k, 0) + int(v)
            for k, v in pd.Series(lab[nouveau]).value_counts().items():
                uniques[k] = uniques.get(k, 0) + int(v)
        print(f"[chargement] passe 1 : {chemin.stem} analysé")
    del vus

    proba = {}
    for k, n in uniques.items():
        p = 1.0
        if k == "Benign" and frac_benign is not None:
            p = frac_benign
        if max_par_classe is not None:
            p = min(p, max_par_classe / n)
        proba[k] = p

    # ---- passe 2 : lecture des lignes gardées
    rng = np.random.default_rng(graine)
    morceaux, i = [], 0
    for chemin in fichiers:
        for lot in pq.ParquetFile(chemin).iter_batches(batch_size=taille_lot):
            df = lot.to_pandas()
            df = df.drop(columns=[c for c in config.COLONNES_FUITE if c in df.columns])
            lab = df[config.CIBLE].astype(str)
            p = lab.map(proba).to_numpy(dtype="float64")
            garder = masques[i] & (rng.random(len(df)) < p)
            morceaux.append(df.loc[garder])
            i += 1
        print(f"[chargement] passe 2 : {chemin.stem} lu")
    df = pd.concat(morceaux, ignore_index=True)
    del morceaux

    total_brut, total_unique = sum(bruts.values()), sum(uniques.values())
    df.attrs["chargement"] = {
        "lignes_brutes": total_brut, "lignes_uniques": total_unique,
        "doublons_retires": total_brut - total_unique,
        "par_classe": pd.DataFrame({"brutes": pd.Series(bruts), "uniques": pd.Series(uniques),
                                    "probabilite_tirage": pd.Series(proba)}),
    }
    print(f"[chargement] {total_brut:,} lignes, dont {total_brut - total_unique:,} doublons ; "
          f"{len(df):,} lignes gardées ({df.memory_usage().sum() / 1e6:,.0f} Mo)")
    return df
