"""Point d'entrée en ligne de commande de la partie 2.

Exemples (Windows PowerShell ou terminal Linux/macOS) :
    python main.py convertir
    python main.py audit-complet
    python main.py audit --frac-benign 0.05
    python main.py nettoyer --frac-benign 0.1 --max-par-classe 300000
    python main.py decouper
    python main.py entrainer --modele xgboost --auteur IA2
    python main.py entrainer --modele tous
    python main.py optimiser --modele lightgbm --essais 30
    python main.py expliquer --modele xgboost
    python main.py deep --architecture mlp

Option globale --donnees : dossier des données (par défaut « data »), avec les
sous-dossiers raw/ (CSV), parquet/, clean/ et splits/.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src import config  # noqa: E402


def chemins(args):
    d = Path(args.donnees)
    r = Path(args.resultats)
    p = {"csv": d / "raw", "parquet": d / "parquet", "propre": d / "clean",
         "splits": d / "splits", "resultats": r, "figures": r / "figures",
         "modeles": Path(args.modeles)}
    for k, v in p.items():
        if k != "csv":
            v.mkdir(parents=True, exist_ok=True)
    from src import evaluation
    evaluation.FICHIER_RESULTATS = r / "resultats.csv"
    return p


def cmd_convertir(a, p):
    from src.chargement import convertir_csv_en_parquet
    convertir_csv_en_parquet(p["csv"], p["parquet"], taille_morceau=a.taille_morceau)


def cmd_audit_complet(a, p):
    from src.audit import audit_complet
    audit_complet(p["parquet"], sortie=p["resultats"] / "audit_complet")


def cmd_audit(a, p):
    from src.audit import rapport_audit
    from src.chargement import charger
    df = charger(p["parquet"], frac_benign=a.frac_benign, max_par_classe=a.max_par_classe)
    print(f"Audit sur {len(df):,} lignes")
    rapport_audit(df, p["parquet"], sortie=p["resultats"] / "audit")


def cmd_nettoyer(a, p):
    from src.chargement import charger_pour_nettoyage
    from src.nettoyage import nettoyer, sauvegarder
    df = charger_pour_nettoyage(p["parquet"], frac_benign=a.frac_benign,
                                max_par_classe=a.max_par_classe)
    print(df.attrs["chargement"]["par_classe"].to_string())
    propre, journal = nettoyer(df, regrouper_web=a.regrouper_web,
                               min_par_classe=a.min_par_classe)
    del df
    print(journal.bilan_classes(propre).to_string())
    sauvegarder(propre, journal, p["propre"])


def cmd_decouper(a, p):
    from src.nettoyage import charger_propre
    from src.pretraitement import decouper, sauvegarder_splits
    df = charger_propre(p["propre"])
    sauvegarder_splits(*decouper(df), dossier=p["splits"])


FENETRES = ["Init Fwd Win Byts", "Init Bwd Win Byts"]


def _transformer_colonnes(a, *Xs):
    """Expériences de robustesse : retirer des colonnes, ou réduire la fenêtre TCP
    à une information générique (fenêtre observée ou non)."""
    sortie = []
    for X in Xs:
        X = X.copy()
        if getattr(a, "binariser_fenetre", False):
            for c in FENETRES:
                if c in X:
                    X[c + " observee"] = (X[c] >= 0).astype("float32")
                    X = X.drop(columns=[c])
        a_retirer = [c for c in (getattr(a, "exclure", None) or []) if c in X]
        sortie.append(X.drop(columns=a_retirer))
    return sortie


def _nom(a, nom):
    return f"{nom}_{a.suffixe}" if getattr(a, "suffixe", "") else nom


def _donnees_entrainement(a, p):
    from src.pretraitement import charger_splits, reequilibrer
    X_train, X_val, X_test, y_train, y_val, y_test, enc = charger_splits(p["splits"])
    inconnues = [c for c in (a.exclure or []) if c not in X_train.columns]
    if inconnues:
        raise SystemExit(f"Colonnes inconnues : {inconnues}. Disponibles : {list(X_train.columns)}")
    X_train, X_val, X_test = _transformer_colonnes(a, X_train, X_val, X_test)
    if a.exclure or a.binariser_fenetre:
        print(f"[colonnes] {X_train.shape[1]} caractéristiques utilisées")
    if not a.sans_reequilibrage:
        X_train, y_train = reequilibrer(X_train, y_train, plafond=a.plafond, plancher=a.plancher)
        print(f"[rééquilibrage] train : {len(X_train):,} lignes après sous-échantillonnage + SMOTE")
    return X_train, X_val, X_test, y_train, y_val, y_test, enc


def cmd_entrainer(a, p):
    from src import modeles
    from src.evaluation import evaluer, principales_confusions, tableau_comparatif
    X_train, X_val, X_test, y_train, y_val, y_test, enc = _donnees_entrainement(a, p)
    noms = list(modeles.CATALOGUE) if a.modele == "tous" else [a.modele]
    if a.modele == "tous":
        noms.remove("knn")  # trop lent sur le jeu complet ; à lancer à part sur un échantillon
    for algo in noms:
        nom = _nom(a, algo)
        m, duree = modeles.entrainer(algo, X_train, y_train)
        evaluer(m, X_val, y_val, enc, nom, "val", auteur=a.auteur, dossier_figures=p["figures"])
        r = evaluer(m, X_test, y_test, enc, nom, "test", auteur=a.auteur,
                    params={"entrainement_s": round(duree, 1), "plafond": a.plafond,
                            "exclure": a.exclure, "binariser_fenetre": a.binariser_fenetre},
                    dossier_figures=p["figures"])
        print("\nPrincipales confusions :\n", principales_confusions(y_test, r["y_pred"], enc))
        print("Modèle sauvegardé :", modeles.sauvegarder_modele(m, nom, p["modeles"]))
    print("\n=== Tableau comparatif (test) ===\n", tableau_comparatif().to_string(index=False))


def cmd_optimiser(a, p):
    from src import modeles
    from src.evaluation import evaluer
    X_train, X_val, X_test, y_train, y_val, y_test, enc = _donnees_entrainement(a, p)
    etude = modeles.optimiser(a.modele, X_train, y_train, X_val, y_val, n_essais=a.essais)
    etude.trials_dataframe().to_csv(p["resultats"] / f"optuna_{a.modele}.csv", index=False)
    m, _ = modeles.entrainer(a.modele, X_train, y_train, etude.best_params)
    nom = _nom(a, f"{a.modele}_optuna")
    evaluer(m, X_test, y_test, enc, nom, "test", params=etude.best_params, auteur=a.auteur,
            dossier_figures=p["figures"])
    print("Modèle sauvegardé :", modeles.sauvegarder_modele(m, nom, p["modeles"]))


def cmd_expliquer(a, p):
    from src import explicabilite as ex
    from src.modeles import charger_modele
    from src.pretraitement import charger_splits
    _, _, X_test, _, _, _, enc = charger_splits(p["splits"])
    m = charger_modele(a.modele, p["modeles"])
    attendues = list(getattr(m, "feature_names_in_", X_test.columns))
    if any(c.endswith(" observee") for c in attendues):
        for c in FENETRES:
            X_test[c + " observee"] = (X_test[c] >= 0).astype("float32")
    X_test = X_test[attendues]  # mêmes colonnes que pendant l'entraînement
    expl, _ = ex.valeurs_shap(m, X_test, n_max=a.n)
    imp = ex.importance_shap_par_classe(expl, enc)
    imp.to_csv(p["resultats"] / f"shap_{a.modele}.csv")
    print(imp.head(15).round(4).to_string())
    classes = a.classes or [c for c in enc.classes_ if c != "Benign"][:4]
    for c in ex.tracer_shap(expl, enc, a.modele, classes=classes, dossier=p["figures"]):
        print("Figure :", c)


def cmd_deep(a, p):
    from src.deep_learning import ModeleKeras
    from src.evaluation import evaluer
    X_train, X_val, X_test, y_train, y_val, y_test, enc = _donnees_entrainement(a, p)
    m = ModeleKeras(a.architecture, epoques=a.epoques).fit(X_train, y_train, X_val, y_val)
    nom = _nom(a, f"keras_{a.architecture}")
    evaluer(m, X_test, y_test, enc, nom, "test", auteur=a.auteur, dossier_figures=p["figures"])
    m.reseau.save(Path(p["modeles"]) / f"{nom}.keras")


def cmd_robustesse(a, p):
    from src import robustesse as rb
    from src.nettoyage import charger_propre
    X, y, jours = rb.donnees_avec_jour(charger_propre(p["propre"]))
    print(f"{len(X):,} flux, {jours.nunique()} jours de capture")
    dossier = p["resultats"] / "robustesse"
    dossier.mkdir(parents=True, exist_ok=True)
    if a.mode == "binaire":
        t = rb.jours_exclus_binaire(X, y, jours, a.variantes, a.plafond, a.arbres)
        r = rb.resume(t, ["rappel_attaques", "taux_fausses_alertes", "f1", "roc_auc"])
    else:
        t = rb.jours_exclus_multiclasse(X, y, jours, a.variantes, a.plafond, a.arbres)
        r = rb.resume(t, ["f1_macro_classes_du_jour"]) if not t.empty else t
    t.to_csv(dossier / f"jours_exclus_{a.mode}.csv", index=False)
    r.to_csv(dossier / f"resume_{a.mode}.csv")
    print(f"\n=== Moyenne et écart type sur les jours exclus ({a.mode}) ===\n{r.to_string()}")


def cmd_comparer_shap(a, p):
    from src.modeles import charger_modele
    from src.pretraitement import charger_splits
    from src.robustesse import comparer_shap
    _, _, X_test, _, _, _, _ = charger_splits(p["splits"])
    t = comparer_shap({n: charger_modele(n, p["modeles"]) for n in a.noms}, X_test, n_max=a.n)
    t.to_csv(p["resultats"] / "shap_comparaison.csv")
    print(t.to_string())


def main():
    ap = argparse.ArgumentParser(description="Partie 2 : classification des attaques (CSE-CIC-IDS2018)")
    ap.add_argument("--donnees", default=str(config.RACINE / "data"))
    ap.add_argument("--resultats", default=str(config.DOSSIER_RESULTATS))
    ap.add_argument("--modeles", default=str(config.DOSSIER_MODELES))
    sp = ap.add_subparsers(dest="commande", required=True)

    s = sp.add_parser("convertir", help="CSV -> Parquet (une seule fois)")
    s.add_argument("--taille-morceau", type=int, default=500_000)
    s.set_defaults(f=cmd_convertir)

    sp.add_parser("audit-complet", help="comptages exacts sur tout le jeu").set_defaults(f=cmd_audit_complet)

    for nom, f, aide in [("audit", cmd_audit, "les 16 contrôles qualité, sans rien modifier"),
                         ("nettoyer", cmd_nettoyer, "nettoyage + journal")]:
        s = sp.add_parser(nom, help=aide)
        s.add_argument("--frac-benign", type=float, default=None,
                       help="fraction de Benign gardée (ex. 0.1) ; par défaut tout")
        s.add_argument("--max-par-classe", type=int, default=None)
        if nom == "nettoyer":
            s.add_argument("--regrouper-web", action="store_true")
            s.add_argument("--min-par-classe", type=int, default=50,
                           help="classes plus petites après nettoyage retirées (0 = aucune)")
        s.set_defaults(f=f)

    sp.add_parser("decouper", help="split stratifié 70/15/15").set_defaults(f=cmd_decouper)

    def options_entrainement(s):
        s.add_argument("--plafond", type=int, default=200_000, help="max de lignes par classe dans le train")
        s.add_argument("--plancher", type=int, default=5_000, help="min par classe via SMOTE")
        s.add_argument("--sans-reequilibrage", action="store_true")
        s.add_argument("--auteur", default="")
        s.add_argument("--exclure", nargs="*", default=None,
                       help='colonnes retirées, ex. --exclure "Init Fwd Win Byts" "Dst Port"')
        s.add_argument("--binariser-fenetre", action="store_true",
                       help="remplace Init Fwd/Bwd Win Byts par « fenêtre observée : 0/1 »")
        s.add_argument("--suffixe", default="", help="ajouté au nom du modèle, ex. sans_fenetre")

    from src.modeles import CATALOGUE
    s = sp.add_parser("entrainer", help="entraîne et évalue un modèle (ou « tous »)")
    s.add_argument("--modele", required=True, choices=list(CATALOGUE) + ["tous"])
    options_entrainement(s)
    s.set_defaults(f=cmd_entrainer)

    s = sp.add_parser("optimiser", help="recherche d'hyperparamètres avec Optuna")
    s.add_argument("--modele", required=True, choices=["xgboost", "lightgbm", "random_forest"])
    s.add_argument("--essais", type=int, default=30)
    options_entrainement(s)
    s.set_defaults(f=cmd_optimiser)

    s = sp.add_parser("expliquer", help="SHAP sur un modèle sauvegardé (à arbres)")
    s.add_argument("--modele", required=True, help="nom du fichier dans modeles/, sans .joblib")
    s.add_argument("--n", type=int, default=2000, help="nombre de flux expliqués")
    s.add_argument("--classes", nargs="*", default=None)
    s.set_defaults(f=cmd_expliquer)

    s = sp.add_parser("deep", help="réseau de neurones Keras (MLP ou CNN 1D)")
    s.add_argument("--architecture", choices=["mlp", "cnn1d"], default="mlp")
    s.add_argument("--epoques", type=int, default=30)
    options_entrainement(s)
    s.set_defaults(f=cmd_deep)

    from src.robustesse import VARIANTES
    s = sp.add_parser("robustesse", help="test sur des jours de capture jamais vus (E4)")
    s.add_argument("--mode", choices=["binaire", "multiclasse"], default="binaire")
    s.add_argument("--variantes", nargs="+", choices=list(VARIANTES),
                   default=["complet", "sans_fwd_win", "sans_fenetres"])
    s.add_argument("--plafond", type=int, default=50_000, help="max de lignes par classe dans chaque train")
    s.add_argument("--arbres", type=int, default=200)
    s.set_defaults(f=cmd_robustesse)

    s = sp.add_parser("comparer-shap", help="top 10 SHAP de plusieurs modèles sur les mêmes flux (E5)")
    s.add_argument("--noms", nargs="+", required=True, help="noms des modèles dans modeles/")
    s.add_argument("--n", type=int, default=2000)
    s.set_defaults(f=cmd_comparer_shap)

    a = ap.parse_args()
    a.f(a, chemins(a))


if __name__ == "__main__":
    main()
