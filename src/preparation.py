"""
Préparation d'un nouveau client pour le modèle LightGBM.

Reproduit à l'identique le prétraitement de 02_pretraitement.ipynb.
Toute modification ici doit être répercutée là-bas, et inversement.
"""

import json
import re

import numpy as np
import pandas as pd


def nettoyer_nom(nom):
    """
    Reproduit le nettoyage que LightGBM applique aux noms de colonnes :
    tout caractère qui n'est ni une lettre, ni un chiffre, ni « _ » est
    remplacé par « _ ».  « Not Verified » devient « Not_Verified ».
    """
    return re.sub(r"[^\w]+", "_", str(nom))


# ---------------------------------------------------------------
# Tables de correspondance (étape 2 du prétraitement)
# ---------------------------------------------------------------

EMP_LENGTH_MAP = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
}

LETTRES_GRADE = "ABCDEFG"

# codes « non calculable » du ratio d'endettement (étape 4)
CODES_DTI_ABSENCE = {999, -1}

CATEGORIELLES = ["purpose", "home_ownership", "verification_status", "addr_state"]

# colonnes d'entrée consommées par une transformation, donc jamais recopiées telles quelles
CONSOMMEES = {"term", "emp_length", "sub_grade", "issue_d", "earliest_cr_line"}


def sub_grade_vers_num(sub_grade):
    """A1 -> 1, A5 -> 5, B1 -> 6, ... G5 -> 35."""
    lettre = str(sub_grade)[0].upper()
    chiffre = int(str(sub_grade)[1])
    return LETTRES_GRADE.index(lettre) * 5 + chiffre


def mois_entre(debut, fin):
    """Nombre de mois entiers entre deux dates."""
    d, f = pd.to_datetime(debut), pd.to_datetime(fin)
    return (f.year - d.year) * 12 + (f.month - d.month)


def charger_artefacts(chemin="../models/preprocessing_artefacts.json"):
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------

def preparer_client(client, artefacts, colonnes_modele, strict=True):
    """
    client          : dict des valeurs brutes du client
    artefacts       : dict chargé depuis preprocessing_artefacts.json
    colonnes_modele : modele.feature_name_ (les 107 noms, dans l'ordre)
    strict          : si True, lève une erreur quand une variable
                      obligatoire manque, au lieu de la mettre à 0

    Retourne un DataFrame d'une ligne, 107 colonnes, prêt pour predict_proba.
    """
    c = dict(client)          # copie : on ne modifie pas l'original
    ligne = {}

    # --- 1. durée du prêt : "36 months" -> 36
    ligne["term_clean"] = int(str(c["term"]).split()[0])

    # --- 2. ancienneté dans l'emploi : indicateur AVANT conversion
    emp = c.get("emp_length")
    absent = emp is None or (isinstance(emp, float) and np.isnan(emp)) or emp == ""
    ligne["emp_length_manquant"] = 1 if absent else 0
    ligne["emp_length_num"] = np.nan if absent else EMP_LENGTH_MAP[emp]

    # --- 3. sous-catégorie de risque : A1..G5 -> 1..35
    ligne["sub_grade_num"] = sub_grade_vers_num(c["sub_grade"])

    # --- 4. ancienneté du dossier de crédit, vue le jour de l'octroi
    ligne["credit_age_mois"] = mois_entre(c["earliest_cr_line"], c["issue_d"])

    # --- 5. demandes de crédit récentes : indicateur d'absence
    # L'indicateur suit mths_since_recent_inq, et non inq_last_6mths :
    # quand aucune demande n'a eu lieu, il n'y a pas de délai à mesurer
    # (étape 6.5 du prétraitement, 103 155 prêts concernés).
    mths_inq = c.get("mths_since_recent_inq")
    absent_inq = mths_inq is None or (isinstance(mths_inq, float) and np.isnan(mths_inq))
    ligne["inq_aucune"] = 1 if absent_inq else 0
    ligne["mths_since_recent_inq"] = np.nan if absent_inq else mths_inq

    # --- 5 bis. ratio d'endettement : 999 et -1 sont des codes d'absence
    dti = c.get("dti")
    if dti is not None and float(dti) in CODES_DTI_ABSENCE:
        c["dti"] = np.nan

    # --- 6. variables numériques restantes, recopiées telles quelles
    for k, v in c.items():
        if k not in CONSOMMEES and k not in CATEGORIELLES and k not in ligne:
            ligne[k] = v

    # --- 7. encodage des variables catégorielles en colonnes 0/1
    colonnes_actives = []
    for var in CATEGORIELLES:
        valeur = c.get(var)
        rares = artefacts["modalites_rares"].get(var, [])
        if valeur in rares:                                  # modalité rare
            valeur = artefacts["regroupement"][var]          # -> AUTRE / other
        nom = nettoyer_nom(f"{var}_{valeur}")
        if nom not in colonnes_modele:
            raise ValueError(
                f"Modalité inconnue pour {var} : '{c.get(var)}' "
                f"(colonne '{nom}' absente du modèle)"
            )
        ligne[nom] = 1
        colonnes_actives.append(nom)

    # --- 8. contrôle : rien d'obligatoire ne doit manquer
    imputables = set(artefacts["valeurs_imputation"])
    binaires = {col for col in colonnes_modele
                if col.split("_")[0] in {"purpose", "home", "verification", "addr"}}
    manquantes = [col for col in colonnes_modele
                  if col not in ligne and col not in imputables and col not in binaires]
    if manquantes and strict:
        raise ValueError(
            f"{len(manquantes)} variable(s) obligatoire(s) absente(s) : "
            f"{manquantes[:10]}{' ...' if len(manquantes) > 10 else ''}"
        )

    # --- 9. assembler dans l'ordre exact attendu par le modèle
    X = pd.DataFrame([{col: ligne.get(col, np.nan) for col in colonnes_modele}])

    # --- 10. imputation, avec les valeurs calculées sur le train
    for col, val in artefacts["valeurs_imputation"].items():
        if col in X.columns:
            X[col] = X[col].fillna(val)

    # --- 11. les colonnes 0/1 non activées valent 0
    X[list(binaires)] = X[list(binaires)].fillna(0)

    X = X.astype(float)
    return X[colonnes_modele]