"""
API d'inférence du modèle de scoring.
Lancement (depuis le dossier src) :  uvicorn api:app --port 8000
Documentation interactive :          http://127.0.0.1:8000/docs
Routes : /  (accueil)  ·  /sante  ·  /modele  ·  /modalites  ·  /predire
"""
import datetime as dt
import json
import os
from typing import Literal, Optional

import joblib
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from preparation import charger_artefacts, preparer_client

# ---------------------------------------------------------------
# 1. Chargement du modèle et des artefacts (une seule fois, au démarrage)
# ---------------------------------------------------------------
ICI = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ICI, "..", "models")
FICHIER_MODELE = "lightgbm_final.joblib"     # même modèle que l'interface Streamlit

modele = joblib.load(os.path.join(MODELS, FICHIER_MODELE))
artefacts = charger_artefacts(os.path.join(MODELS, "preprocessing_artefacts.json"))
with open(os.path.join(MODELS, "valeurs_par_defaut.json"), encoding="utf-8") as f:
    defauts = json.load(f)
explainer = shap.TreeExplainer(modele)
colonnes = list(modele.feature_name_)


def modalites(prefixe):
    """Valeurs acceptées pour une variable catégorielle (lues dans les colonnes du modèle)."""
    n = len(prefixe) + 1
    return sorted(c[n:] for c in colonnes if c.startswith(prefixe + "_"))


MODALITES = {
    "purpose": modalites("purpose"),
    "home_ownership": modalites("home_ownership"),
    "addr_state": modalites("addr_state"),
    "verification_status": [m.replace("_", " ") for m in modalites("verification_status")],
}

# ---------------------------------------------------------------
# 2. Le format d'un dossier (ce que l'API accepte en entrée)
# ---------------------------------------------------------------
class Dossier(BaseModel):
    loan_amnt: float = Field(..., ge=500, le=40000, description="Montant demandé ($)")
    term: Literal["36 months", "60 months"] = Field(..., description="Durée du prêt")
    int_rate: float = Field(..., ge=5, le=31, description="Taux d'intérêt (%)")
    sub_grade: str = Field(..., pattern=r"^[A-G][1-5]$", description="Note LendingClub (A1 à G5)")
    purpose: str = Field(..., description="Objet du financement")
    annual_inc: float = Field(..., ge=0, description="Revenu annuel ($)")
    verification_status: str = Field(..., description="Vérification du revenu")
    dti: float = Field(..., ge=0, le=100, description="Ratio d'endettement (%)")
    emp_length: Optional[str] = Field(None, description="Ancienneté dans l'emploi, ex. '5 years' (vide si inconnue)")
    home_ownership: str = Field(..., description="Logement")
    addr_state: str = Field(..., description="État de résidence (code à 2 lettres)")
    fico_range_low: int = Field(..., ge=600, le=850, description="Score FICO (borne basse)")
    acc_open_past_24mths: int = Field(..., ge=0, description="Comptes ouverts sur 24 mois")
    inq_last_6mths: int = Field(0, ge=0, description="Demandes de crédit sur 6 mois")
    mths_since_recent_inq: Optional[int] = Field(None, ge=0, description="Mois depuis la dernière demande (vide si aucune)")
    earliest_cr_line: dt.date = Field(..., description="Date du premier crédit (AAAA-MM-JJ)")
    seuil: float = Field(0.5, ge=0.05, le=0.95, description="Seuil de refus")

    model_config = {"json_schema_extra": {"examples": [{
        "loan_amnt": 12000, "term": "36 months", "int_rate": 13.5, "sub_grade": "C2",
        "purpose": "debt_consolidation", "annual_inc": 55000, "verification_status": "Source Verified",
        "dti": 18.5, "emp_length": "5 years", "home_ownership": "RENT", "addr_state": "CA",
        "fico_range_low": 690, "acc_open_past_24mths": 4, "inq_last_6mths": 1,
        "mths_since_recent_inq": 3, "earliest_cr_line": "2008-06-01", "seuil": 0.5}]}}


# ---------------------------------------------------------------
# 3. L'application et ses routes
# ---------------------------------------------------------------
app = FastAPI(
    title="API de scoring du risque de crédit",
    description="Reçoit un dossier de prêt, renvoie un score de risque, une décision suggérée "
                "et les principaux facteurs (SHAP). Outil d'aide à la décision : la décision finale "
                "reste humaine.",
    version="1.0",
)


@app.get("/", include_in_schema=False)
def accueil():
    """Page d'accueil : renvoie vers la documentation interactive."""
    return RedirectResponse(url="/docs")


@app.get("/sante")
def sante():
    """Vérifie que l'API fonctionne et indique le modèle chargé."""
    return {"statut": "ok", "modele": FICHIER_MODELE, "nombre_de_variables": len(colonnes)}


@app.get("/modalites")
def lister_modalites():
    """Valeurs acceptées pour les variables catégorielles."""
    return MODALITES


@app.get("/modele")
def fiche_modele():
    """Fiche d'identité du modèle servi : traçabilité de la décision."""
    return {
        "algorithme": "LightGBM (gradient boosting)",
        "fichier": FICHIER_MODELE,
        "periode_entrainement": "octobre 2012 à décembre 2016",
        "reglages": {"n_estimators": 432, "learning_rate": 0.05,
                     "num_leaves": 63, "min_child_samples": 100,
                     "class_weight": "balanced"},
        "performances_test_2017": {"auc": 0.7224, "rappel": 0.689,
                                   "precision": 0.360, "f1": 0.473},
        "nombre_de_variables": len(colonnes),
        "explication": "SHAP (TreeExplainer), contributions par dossier",
        "suivi": "Expériences et versions enregistrées dans MLflow (scoring_credit_lightgbm, version 1)",
    }


@app.post("/predire")
def predire(dossier: Dossier):
    """Évalue un dossier : score, décision suggérée et 5 principaux facteurs."""
    # a) contrôle des variables catégorielles
    for champ, valeurs in MODALITES.items():
        if getattr(dossier, champ) not in valeurs:
            raise HTTPException(422, f"Valeur non reconnue pour {champ}. Voir la route /modalites.")

    # b) même préparation que l'interface : valeurs par défaut, puis valeurs saisies
    client = dict(defauts)
    client.update(dossier.model_dump(exclude={"seuil"}))
    client["earliest_cr_line"] = dossier.earliest_cr_line.isoformat()
    client["issue_d"] = dt.date.today().isoformat()
    try:
        X = preparer_client(client, artefacts, colonnes)
    except ValueError as e:
        raise HTTPException(422, f"Dossier incomplet : {e}")

    # c) score et décision
    score = float(modele.predict_proba(X)[:, 1][0])
    refuse = score >= dossier.seuil

    # d) explication SHAP : les 5 variables qui pèsent le plus pour ce dossier
    valeurs = explainer.shap_values(X)
    if isinstance(valeurs, list):
        valeurs = valeurs[-1]
    valeurs = np.asarray(valeurs)
    if valeurs.ndim == 3:
        valeurs = valeurs[:, :, -1]
    v = valeurs[0]
    principaux = np.argsort(-np.abs(v))[:5]
    facteurs = [{"variable": colonnes[i],
                 "contribution": round(float(v[i]), 4),
                 "effet": "augmente le risque" if v[i] > 0 else "diminue le risque"}
                for i in principaux]

    return {
        "score": round(score, 4),
        "seuil": dossier.seuil,
        "decision_suggeree": "refus" if refuse else "accord",
        "facteurs_principaux": facteurs,
        "avertissement": "Score non calibré (pondération des classes) : à lire comme un classement du risque, pas comme une probabilité.",
        "modele": FICHIER_MODELE,
    }