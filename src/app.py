"""
Interface de scoring du risque de crédit.

Parcours en quatre écrans :
    accueil  ->  identité du demandeur  ->  dossier financier  ->  résultat

Lancement, depuis la racine du projet :
    streamlit run src/app.py
"""

import datetime as dt
import io
import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st
import streamlit.components.v1 as components
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preparation import charger_artefacts, preparer_client  # noqa: E402

ICI = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ICI, "..", "models")


# ===============================================================
# 1. CONFIGURATION DE LA PAGE
#    Doit rester la toute première commande Streamlit.
# ===============================================================
st.set_page_config(
    page_title="Scoring du risque de crédit",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===============================================================
# 2. FEUILLE DE STYLE
#    Palette reprise du document « Mon mémoire expliqué » :
#    encre marine, bleu pétrole en accent, fond gris très clair.
#    Aucune balise <h1>…<h4> n'est utilisée dans la page :
#    Streamlit leur ajoute une icône de lien au survol.
# ===============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root{
  --fond:#F7F8FA; --carte:#FFFFFF; --encre:#141B26; --encre-douce:#4C5768;
  --trait:#D5DCE4; --trait-doux:#EAEEF3;
  --accent:#0F6E6E; --accent-doux:#E2EFEE; --accent-fonce:#0B4F4F; --accent-clair:#7FD6CF;
  --vert:#2E7D5B; --vert-doux:#E6F2EC; --rouge:#A8352E; --rouge-doux:#F8E7E5;
  --titre:'Fraunces', Georgia, 'Times New Roman', serif;
  --texte:'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
  --chiffre:'IBM Plex Mono', Consolas, 'Courier New', monospace;
}

/* ---------- Base : fond, police, taille des textes ---------- */
.stApp{background:var(--fond);color:var(--encre);font-family:var(--texte)}
.stApp p, .stApp li{font-size:1.06rem;line-height:1.6}

/* ---------- Le grand blanc en haut de page ---------- */
[data-testid="stHeader"]{background:transparent}
[data-testid="stAppDeployButton"]{display:none}
.block-container, [data-testid="stMainBlockContainer"]{
  padding-top:2.4rem !important; padding-bottom:3rem !important; max-width:1280px;
}

/* ---------- Libellés des champs ---------- */
[data-testid="stWidgetLabel"] p{
  font-size:1.02rem !important; font-weight:600 !important; color:var(--encre) !important;
}
.stCheckbox p{font-size:1.04rem !important; font-weight:600 !important}

/* ---------- Champs : plus hauts, texte plus grand ---------- */
.stTextInput input, .stNumberInput input, .stDateInput input{
  font-size:1.1rem !important; font-family:var(--texte) !important;
  min-height:3.15rem; color:var(--encre) !important;
}
div[data-baseweb="select"] > div{min-height:3.15rem}
div[data-baseweb="select"] div, div[data-baseweb="select"] span, div[data-baseweb="select"] input{
  font-size:1.1rem !important; font-family:var(--texte) !important;
}
[data-baseweb="popover"] li{font-size:1.06rem !important; font-family:var(--texte) !important}
div[data-baseweb="input"], div[data-baseweb="select"] > div,
[data-testid="stNumberInputContainer"]{border-color:var(--trait) !important; border-radius:9px !important}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within{
  border-color:var(--accent) !important;
}

/* ---------- Boutons, sans icônes ---------- */
.stButton > button, [data-testid="stDownloadButton"] > button{
  min-height:3.15rem; border-radius:9px; padding:0 1.1rem; font-family:var(--texte);
}
.stButton > button p, [data-testid="stDownloadButton"] button p{
  font-size:1.06rem !important; font-weight:600 !important;
}
button[kind="primary"], [data-testid="stBaseButton-primary"]{
  background:var(--accent) !important; border:1px solid var(--accent) !important;
}
button[kind="primary"] p, [data-testid="stBaseButton-primary"] p{color:#FFFFFF !important}
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover{
  background:var(--accent-fonce) !important; border-color:var(--accent-fonce) !important;
}
button[kind="secondary"], [data-testid="stBaseButton-secondary"]{
  background:var(--carte) !important; border:1.5px solid var(--trait) !important;
}
button[kind="secondary"] p, [data-testid="stBaseButton-secondary"] p{color:var(--encre) !important}
button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover{
  border-color:var(--accent) !important;
}
[data-testid="stDownloadButton"] button{
  background:var(--encre) !important; border:1px solid var(--encre) !important;
}
[data-testid="stDownloadButton"] button p{color:#FFFFFF !important}

/* ---------- Cartes (conteneurs avec bordure) ---------- */
[class*="st-key-carte"]{
  background:var(--carte); border-radius:12px; padding:1.5rem 1.7rem 1.3rem;
  box-shadow:0 1px 2px rgba(20,27,38,.04), 0 6px 20px rgba(20,27,38,.05);
}

/* ---------- Bandeau du haut ---------- */
.bandeau{display:flex;justify-content:space-between;align-items:center;gap:1.4rem;flex-wrap:wrap;
  background:var(--encre);border-radius:12px;padding:1.1rem 1.6rem;margin-bottom:1.5rem}
.bandeau-sur{font-family:var(--chiffre);font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent-clair);font-weight:600}
.bandeau-titre{font-family:var(--titre);font-size:1.55rem;font-weight:600;color:#FFFFFF;
  margin-top:.15rem;line-height:1.2}
.bandeau-info{font-family:var(--chiffre);font-size:.82rem;color:#AEB9C7;text-align:right;line-height:1.6}

/* ---------- Indicateur d'étapes ---------- */
.etapes{display:flex;align-items:center;gap:.8rem;margin:0 0 1.5rem;flex-wrap:wrap}
.etape{display:flex;align-items:center;gap:.55rem;font-weight:600;font-size:1.02rem;color:var(--encre-douce)}
.etape .n{width:2.1rem;height:2.1rem;border-radius:50%;display:inline-flex;align-items:center;
  justify-content:center;font-family:var(--chiffre);font-size:.92rem;border:1.5px solid var(--trait);
  background:var(--carte);color:var(--encre-douce)}
.etape.actif{color:var(--encre)}
.etape.actif .n{background:var(--accent);border-color:var(--accent);color:#FFFFFF}
.etape.fait .n{background:var(--accent-doux);border-color:var(--accent);color:var(--accent-fonce)}
.etapes .lien{flex:1;min-width:1.5rem;max-width:6rem;height:2px;background:var(--trait)}
.etapes .lien.fait{background:var(--accent)}

/* ---------- Titres de page et de section ---------- */
.titre-page{font-family:var(--titre);font-size:2rem;font-weight:600;line-height:1.2;
  color:var(--encre);margin:0 0 .3rem}
.sous-titre{font-size:1.12rem;color:var(--encre-douce);margin:0 0 1.4rem;line-height:1.55}
.sec-titre{font-family:var(--titre);font-size:1.32rem;font-weight:600;color:var(--encre);line-height:1.25}
.sec-sous{font-size:1rem;color:var(--encre-douce);margin:.2rem 0 .4rem}

/* ---------- Page d'accueil ---------- */
.hero-sur{font-family:var(--chiffre);font-size:.82rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);font-weight:600;margin:.4rem 0 .9rem}
.hero-titre{font-family:var(--titre);font-size:clamp(2.2rem,4.2vw,3.2rem);font-weight:600;
  line-height:1.1;color:var(--encre);margin:0 0 1rem;letter-spacing:-.015em}
.hero-texte{font-size:1.22rem;color:var(--encre-douce);max-width:46rem;line-height:1.6;margin:0}
.grille3{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:2rem 0 1.4rem}
@media (max-width:820px){.grille3{grid-template-columns:1fr}}
.tuile{background:var(--carte);border:1px solid var(--trait-doux);border-radius:12px;padding:1.3rem 1.4rem;
  box-shadow:0 1px 2px rgba(20,27,38,.04)}
.tuile-n{font-family:var(--chiffre);font-size:.88rem;font-weight:600;color:var(--accent);margin-bottom:.55rem}
.tuile-t{font-family:var(--titre);font-size:1.22rem;font-weight:600;color:var(--encre);
  margin-bottom:.4rem;line-height:1.25}
.tuile-d{font-size:1.02rem;color:var(--encre-douce);line-height:1.55}
.chiffre-v{font-family:var(--chiffre);font-size:2rem;font-weight:600;color:var(--encre);line-height:1.1}
.chiffre-l{font-size:1rem;color:var(--encre-douce);margin-top:.4rem;line-height:1.45}
.avis{background:var(--accent-doux);border:1px solid #BFDCD9;border-radius:10px;padding:1rem 1.3rem;
  font-size:1.02rem;color:var(--accent-fonce);line-height:1.55;margin:0 0 1.6rem;max-width:56rem}

/* ---------- Messages ---------- */
.message{border-radius:10px;padding:.9rem 1.2rem;font-size:1.02rem;line-height:1.5;margin:.4rem 0 .8rem}
.message.erreur{background:var(--rouge-doux);color:var(--rouge);border:1px solid #E9C3BF}

/* ---------- Résultat ---------- */
.verdict{border-radius:12px;padding:1.25rem 1.5rem;border:1.5px solid}
.verdict.accord{background:var(--vert-doux);border-color:var(--vert)}
.verdict.refus{background:var(--rouge-doux);border-color:var(--rouge)}
.verdict-sur{font-family:var(--chiffre);font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;font-weight:600}
.verdict-titre{font-family:var(--titre);font-size:1.6rem;font-weight:600;margin:.3rem 0 .25rem;line-height:1.2}
.verdict.accord .verdict-sur, .verdict.accord .verdict-titre{color:var(--vert)}
.verdict.refus .verdict-sur, .verdict.refus .verdict-titre{color:var(--rouge)}
.verdict-detail{font-size:1.04rem;color:var(--encre)}
.score{display:flex;align-items:baseline;gap:.8rem;margin:1.3rem 0 .5rem;flex-wrap:wrap}
.score-v{font-family:var(--chiffre);font-size:2.8rem;font-weight:600;color:var(--encre);line-height:1}
.score-l{font-size:1.04rem;color:var(--encre-douce)}
.jauge{position:relative;height:14px;border-radius:7px;background:var(--trait-doux);margin:.5rem 0 .45rem}
.jauge-plein{position:absolute;left:0;top:0;bottom:0;border-radius:7px}
.jauge-seuil{position:absolute;top:-6px;bottom:-6px;width:3px;background:var(--encre);border-radius:2px}
.jauge-legende{position:relative;height:1.4rem;font-family:var(--chiffre);font-size:.82rem;color:var(--encre-douce)}
.jauge-legende span{position:absolute;top:0;white-space:nowrap}
.note{font-size:.98rem;color:var(--encre-douce);line-height:1.55;margin:.4rem 0 0}
.explication{font-size:1.08rem;line-height:1.65;color:var(--encre)}
.recap{font-size:1.02rem;color:var(--encre-douce);line-height:1.7}
.recap b{color:var(--encre);font-weight:600}

/* ---------- Barre latérale ---------- */
[data-testid="stSidebar"]{background:var(--encre)}
[data-testid="stSidebar"] *{color:#E6EBF1}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{color:#E6EBF1 !important}
[data-testid="stSidebar"] [data-testid="stSliderThumbValue"]{display:none}
.lat-sur{font-family:var(--chiffre);font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent-clair) !important;font-weight:600;margin-top:.4rem}
.lat-titre{font-family:var(--titre);font-size:1.4rem;font-weight:600;color:#FFFFFF !important;margin:.2rem 0 .6rem}
.lat-texte{font-size:1rem;line-height:1.55;color:#C3CCD7 !important;margin-bottom:.9rem}
.lat-seuil{font-family:var(--chiffre);font-size:2.3rem;font-weight:600;color:var(--accent-clair) !important;
  line-height:1;margin:.2rem 0 .1rem}
.tab-seuil{width:100%;border-collapse:collapse;margin-top:.5rem}
.tab-seuil th{font-family:var(--chiffre);font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;
  color:#93A1B2 !important;text-align:right;padding:.45rem .35rem;border-bottom:1px solid #2C3846;font-weight:600}
.tab-seuil td{padding:.55rem .35rem;text-align:right;border-bottom:1px solid #222D39;
  font-family:var(--chiffre);font-size:.95rem}
.tab-seuil th:first-child, .tab-seuil td:first-child{text-align:left}
.tab-seuil tr.actif td{color:var(--accent-clair) !important;font-weight:600}
.lat-note{font-size:.9rem;color:#93A1B2 !important;line-height:1.55;margin-top:.8rem}
</style>
""", unsafe_allow_html=True)


# ===============================================================
# 3. CHARGEMENT DU MODÈLE (une seule fois, mis en cache)
# ===============================================================
@st.cache_resource
def charger():
    modele = joblib.load(os.path.join(MODELS, "lightgbm_final.joblib"))
    artefacts = charger_artefacts(os.path.join(MODELS, "preprocessing_artefacts.json"))
    with open(os.path.join(MODELS, "valeurs_par_defaut.json"), encoding="utf-8") as f:
        defauts = json.load(f)
    explainer = shap.TreeExplainer(modele)
    return modele, artefacts, defauts, explainer


modele, artefacts, defauts, explainer = charger()
colonnes = list(modele.feature_name_)


def modalites(prefixe):
    """Liste des modalités d'une variable encodée en one-hot (ex. purpose_car -> car)."""
    n = len(prefixe) + 1
    return sorted(c[n:] for c in colonnes if c.startswith(prefixe + "_"))


# ===============================================================
# 4. LISTES DE CHOIX ET TRADUCTIONS
#    Les valeurs envoyées au modèle restent en anglais ;
#    seul l'affichage est traduit (format_func).
# ===============================================================
SUB_GRADES = [f"{l}{d}" for l in "ABCDEFG" for d in range(1, 6)]
EMP_LENGTHS = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
               "6 years", "7 years", "8 years", "9 years", "10+ years"]
EMP_FR = {"Non renseignée": "Non renseignée", "< 1 year": "Moins d'un an", "1 year": "1 an",
          "10+ years": "10 ans et plus"}
EMP_FR.update({f"{n} years": f"{n} ans" for n in range(2, 10)})

DUREE_FR = {"36 months": "36 mois", "60 months": "60 mois"}

VERIF = {m: m.replace("_", " ") for m in modalites("verification_status")}
VERIF_FR = {"Not_Verified": "Non vérifié", "Source_Verified": "Vérifié par la source", "Verified": "Vérifié"}

LOGEMENT_FR = {"RENT": "Locataire", "OWN": "Propriétaire", "MORTGAGE": "Propriétaire avec crédit immobilier",
               "OTHER": "Autre", "ANY": "Autre", "NONE": "Non renseigné"}

PURPOSE_FR = {
    "car": "Voiture", "credit_card": "Carte de crédit", "debt_consolidation": "Consolidation de dettes",
    "educational": "Études", "home_improvement": "Amélioration du logement", "house": "Achat immobilier",
    "major_purchase": "Achat important", "medical": "Médical", "moving": "Déménagement",
    "other": "Autre", "renewable_energy": "Énergie renouvelable", "small_business": "Petite entreprise",
    "vacation": "Vacances", "wedding": "Mariage",
}

ETATS_FR = {
    "AK": "Alaska", "AL": "Alabama", "AR": "Arkansas", "AZ": "Arizona", "CA": "Californie",
    "CO": "Colorado", "CT": "Connecticut", "DC": "Washington D.C.", "DE": "Delaware", "FL": "Floride",
    "GA": "Géorgie", "HI": "Hawaï", "IA": "Iowa", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiane", "MA": "Massachusetts", "MD": "Maryland",
    "ME": "Maine", "MI": "Michigan", "MN": "Minnesota", "MO": "Missouri", "MS": "Mississippi",
    "MT": "Montana", "NC": "Caroline du Nord", "ND": "Dakota du Nord", "NE": "Nebraska",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "Nouveau-Mexique", "NV": "Nevada",
    "NY": "New York", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvanie",
    "RI": "Rhode Island", "SC": "Caroline du Sud", "SD": "Dakota du Sud", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VA": "Virginie", "VT": "Vermont", "WA": "Washington",
    "WI": "Wisconsin", "WV": "Virginie-Occidentale", "WY": "Wyoming",
}

TRADUCTIONS = {
    "sub_grade_num": "la note de risque interne", "term_clean": "la durée du prêt",
    "int_rate": "le taux d'intérêt", "loan_amnt": "le montant demandé", "dti": "le ratio d'endettement",
    "fico_range_low": "le score de crédit FICO", "annual_inc": "le revenu annuel",
    "acc_open_past_24mths": "les ouvertures de comptes récentes",
    "mths_since_recent_inq": "l'ancienneté de la dernière demande", "home_ownership_RENT": "le statut locataire",
    "home_ownership_MORTGAGE": "le statut hypothécaire", "home_ownership_OWN": "le statut propriétaire",
    "addr_state_CA": "l'état de résidence (Californie)", "mo_sin_old_rev_tl_op": "l'ancienneté du crédit revolving",
    "inq_last_6mths": "les demandes de crédit récentes", "emp_length_num": "l'ancienneté dans l'emploi",
    "credit_age_mois": "l'âge de l'historique de crédit", "inq_aucune": "l'absence de demande de crédit récente",
}


def libelle(col):
    """Nom court en français d'une variable, pour le graphique SHAP."""
    if col in TRADUCTIONS:
        texte = TRADUCTIONS[col]
        for article in ("la ", "le ", "les ", "l'"):
            if texte.startswith(article):
                texte = texte[len(article):]
                break
        return texte[0].upper() + texte[1:]
    for prefixe, nom, table in (("addr_state_", "État", ETATS_FR), ("home_ownership_", "Logement", LOGEMENT_FR),
                                ("purpose_", "Objet", PURPOSE_FR), ("verification_status_", "Revenu", VERIF_FR)):
        if col.startswith(prefixe):
            valeur = col[len(prefixe):]
            return f"{nom} : {table.get(valeur, valeur)}"
    return col


def nom_naturel(col):
    """Nom d'une variable dans une phrase : « le ratio d'endettement », « l'objet du prêt (voiture) »…"""
    if col in TRADUCTIONS:
        return TRADUCTIONS[col]
    for prefixe, debut, table in (("verification_status_", "la vérification du revenu", VERIF_FR),
                                  ("purpose_", "l'objet du prêt", PURPOSE_FR),
                                  ("home_ownership_", "le logement", LOGEMENT_FR),
                                  ("addr_state_", "l'État de résidence", ETATS_FR)):
        if col.startswith(prefixe):
            valeur = col[len(prefixe):]
            return f"{debut} ({table.get(valeur, valeur).lower()})"
    return col


def fr(nombre, decimales=2):
    """Nombre au format français : virgule décimale."""
    return f"{nombre:.{decimales}f}".replace(".", ",")


def milliers(nombre, separateur=" "):
    """15000 -> 15 000.

    À l'écran : espace fine insécable (typographie française).
    Dans le PDF : espace normale, car la police Helvetica ne connaît pas
    l'espace fine et l'afficherait comme un carré noir.
    """
    return f"{int(nombre):,}".replace(",", separateur)


# ===============================================================
# 5. MÉMOIRE DU FORMULAIRE
#    Les valeurs saisies sont gardées quand on change d'écran :
#    revenir de « Résultat » à « Dossier » retrouve tout ce qui a été tapé.
# ===============================================================
def premier_ou(liste, prefere):
    return prefere if prefere in liste else liste[0]


VALEURS_INITIALES = {
    "f_nom": "", "f_prenom": "",
    "f_loan_amnt": 15000, "f_term": "36 months", "f_int_rate": 12.5, "f_sub_grade": "B3",
    "f_purpose": premier_ou(modalites("purpose"), "debt_consolidation"),
    "f_annual_inc": 65000, "f_verif": list(VERIF)[0], "f_dti": 17.9, "f_emp_length": "5 years",
    "f_home": premier_ou(modalites("home_ownership"), "RENT"),
    "f_etat": premier_ou(modalites("addr_state"), "CA"),
    "f_fico": 690, "f_premier_credit": dt.date(2003, 1, 1), "f_acc_open": 4,
    "f_aucune_demande": True, "f_inq_6m": 0, "f_mths_inq": 7,
}

for cle, valeur in VALEURS_INITIALES.items():
    if cle not in st.session_state:
        st.session_state[cle] = valeur
    else:
        # Réaffecter la valeur empêche Streamlit de l'effacer quand le champ n'est pas affiché.
        st.session_state[cle] = st.session_state[cle]

if "page" not in st.session_state:
    st.session_state.page = "accueil"
if "erreur_identite" not in st.session_state:
    st.session_state.erreur_identite = False


# ----- Navigation (fonctions appelées par les boutons) -----
def aller(page):
    st.session_state.page = page


def valider_identite():
    if st.session_state.f_nom.strip() and st.session_state.f_prenom.strip():
        st.session_state.erreur_identite = False
        st.session_state.page = "dossier"
    else:
        st.session_state.erreur_identite = True


def nouvelle_evaluation():
    for cle, valeur in VALEURS_INITIALES.items():
        st.session_state[cle] = valeur
    st.session_state.erreur_identite = False
    st.session_state.page = "identite"


def carte(cle):
    """Conteneur encadré. Le paramètre key n'existe qu'à partir de Streamlit 1.39."""
    try:
        return st.container(border=True, key=cle)
    except TypeError:
        return st.container(border=True)


# ===============================================================
# 6. EXPLICATION ET RAPPORT PDF
# ===============================================================
def expliquer_decision(valeurs_shap, colonnes):
    paires = list(zip(colonnes, valeurs_shap))
    favorables = sorted(paires, key=lambda x: x[1])[:3]
    risques = sorted(paires, key=lambda x: x[1], reverse=True)[:1]
    noms_fav = [nom_naturel(c) for c, v in favorables if v < -0.02]
    noms_ris = [nom_naturel(c) for c, v in risques if v > 0.02]

    if noms_fav and noms_ris:
        return (f"Les éléments favorables pour ce dossier sont : {', '.join(noms_fav)}. "
                f"Point de vigilance principal : {noms_ris[0]}.")
    if noms_fav:
        return (f"Ce dossier présente plusieurs éléments favorables : {', '.join(noms_fav)}. "
                "Aucun facteur de risque majeur n'a été identifié.")
    return "Ce dossier présente plusieurs facteurs de risque significatifs sans élément compensatoire fort."


def generer_pdf(score, refuse, seuil, client, valeurs_shap, colonnes, nom_complet=""):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    el = []

    el.append(Paragraph("RAPPORT D'ÉVALUATION DU RISQUE DE CRÉDIT", styles["Title"]))
    el.append(Spacer(1, 12))
    el.append(Paragraph(f"Date : {dt.date.today().strftime('%d/%m/%Y')}", styles["Normal"]))
    if nom_complet.strip():
        el.append(Paragraph(f"Demandeur : {nom_complet}", styles["Normal"]))
    el.append(Spacer(1, 20))

    decision = "DOSSIER À RISQUE — REFUS SUGGÉRÉ" if refuse else "DOSSIER ACCEPTABLE"
    el.append(Paragraph(f"DÉCISION : {decision}", styles["Heading1"]))
    el.append(Paragraph(f"Score de risque : {fr(score, 3)}  |  Seuil retenu : {fr(seuil)}", styles["Normal"]))
    el.append(Spacer(1, 20))

    el.append(Paragraph("PROFIL DU DOSSIER", styles["Heading2"]))
    profil = [
        ["Variable", "Valeur"],
        ["Montant demandé", f"{milliers(client.get('loan_amnt', 0), ' ')} $"],
        ["Durée", DUREE_FR.get(client.get("term"), client.get("term", "-"))],
        ["Taux d'intérêt", f"{fr(client.get('int_rate', 0))} %"],
        ["Note de risque", client.get("sub_grade", "-")],
        ["Revenu annuel", f"{milliers(client.get('annual_inc', 0), ' ')} $"],
        ["Ratio d'endettement", f"{fr(client.get('dti', 0), 1)} %"],
        ["Score FICO", str(client.get("fico_range_low", "-"))],
    ]
    table = Table(profil, colWidths=[250, 200])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F6E6E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F2F5F7"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D5DCE4")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    el.append(table)
    el.append(Spacer(1, 20))

    el.append(Paragraph("SYNTHÈSE DE LA DÉCISION", styles["Heading2"]))
    el.append(Paragraph(expliquer_decision(valeurs_shap, colonnes), styles["Normal"]))
    el.append(Spacer(1, 20))

    # Pas de symbole ⚠ ici : la police Helvetica du PDF ne sait pas l'afficher.
    el.append(Paragraph(
        "Ce rapport est un outil d'aide à la décision fondé sur un modèle LightGBM. "
        "La décision finale reste du ressort du comité de crédit.", styles["Italic"]))
    doc.build(el)
    buffer.seek(0)
    return buffer.read()


# ===============================================================
# 7. BARRE LATÉRALE — LA POLITIQUE DE RISQUE
# ===============================================================
with st.sidebar:
    st.markdown("""
<div class="lat-sur">Politique de risque</div>
<div class="lat-titre">Seuil de refus</div>
<div class="lat-texte">Un dossier dont le score atteint ce seuil est signalé comme à risque.
Plus le seuil est bas, plus la banque est prudente.</div>
""", unsafe_allow_html=True)

    seuil = st.slider("Seuil de refus", 0.05, 0.95, 0.50, 0.05, key="seuil", label_visibility="collapsed")
    st.markdown(f'<div class="lat-seuil">{fr(seuil)}</div>', unsafe_allow_html=True)

    lignes = ""
    for s, rappel, precision in ((0.30, "93 %", "28 %"), (0.50, "69 %", "36 %"), (0.70, "27 %", "49 %")):
        classe = ' class="actif"' if abs(seuil - s) < 1e-9 else ""
        lignes += f"<tr{classe}><td>{fr(s)}</td><td>{rappel}</td><td>{precision}</td></tr>"

    st.markdown(f"""
<table class="tab-seuil">
  <thead><tr><th>Seuil</th><th>Défauts détectés</th><th>Alertes justes</th></tr></thead>
  <tbody>{lignes}</tbody>
</table>
<div class="lat-note">Mesuré sur les prêts de 2017. « Défauts détectés » est le rappel,
« alertes justes » la précision. Baisser le seuil attrape plus de défauts, au prix de plus
de fausses alertes.</div>
""", unsafe_allow_html=True)


# ===============================================================
# 8. BANDEAU ET INDICATEUR D'ÉTAPES
# ===============================================================
st.markdown("""
<div class="bandeau">
  <div>
    <div class="bandeau-sur">Outil d'aide à la décision</div>
    <div class="bandeau-titre">Scoring du risque de crédit</div>
  </div>
  <div class="bandeau-info">Modèle LightGBM réglé<br>69 % des défauts détectés · F1 0,473 (prêts de 2017)</div>
</div>
""", unsafe_allow_html=True)

ORDRE = ["identite", "dossier", "resultat"]
NOMS_ETAPES = {"identite": "Identité", "dossier": "Dossier financier", "resultat": "Résultat"}

if st.session_state.page in ORDRE:
    rang = ORDRE.index(st.session_state.page)
    morceaux = []
    for i, p in enumerate(ORDRE):
        etat = "actif" if i == rang else ("fait" if i < rang else "")
        if i > 0:
            morceaux.append(f'<span class="lien {"fait" if i <= rang else ""}"></span>')
        morceaux.append(f'<div class="etape {etat}"><span class="n">{i + 1}</span>{NOMS_ETAPES[p]}</div>')
    st.markdown(f'<div class="etapes">{"".join(morceaux)}</div>', unsafe_allow_html=True)


# ===============================================================
# 9. ÉCRAN 0 — ACCUEIL
# ===============================================================
if st.session_state.page == "accueil":
    st.markdown("""
<div class="hero-sur">Gestion adaptative du risque de crédit</div>
<div class="hero-titre">Bienvenue sur votre outil<br>d'évaluation du risque de crédit</div>
<div class="hero-texte">Évaluez en quelques minutes le risque de défaut d'une demande de prêt,
et comprenez sur quels éléments du dossier repose la décision.</div>
<div style="height:1.5rem"></div>
""", unsafe_allow_html=True)

    col_bouton, _ = st.columns([1, 2])
    with col_bouton:
        st.button("Commencer une évaluation", type="primary", use_container_width=True,
                  on_click=aller, args=("identite",), key="btn_commencer")

    st.markdown("""
<div class="grille3">
  <div class="tuile">
    <div class="tuile-n">Étape 1</div>
    <div class="tuile-t">Identifier le demandeur</div>
    <div class="tuile-d">Nom et prénom, repris sur le rapport final.</div>
  </div>
  <div class="tuile">
    <div class="tuile-n">Étape 2</div>
    <div class="tuile-t">Saisir le dossier</div>
    <div class="tuile-d">Le prêt demandé, le profil de l'emprunteur et son historique de crédit.</div>
  </div>
  <div class="tuile">
    <div class="tuile-n">Étape 3</div>
    <div class="tuile-t">Lire la décision expliquée</div>
    <div class="tuile-d">Un score, une décision suggérée, les facteurs qui l'ont emportée,
      et un rapport PDF à joindre au dossier.</div>
  </div>
</div>

<div class="grille3" style="margin-top:0">
  <div class="tuile">
    <div class="chiffre-v">69 %</div>
    <div class="chiffre-l">des défauts détectés au seuil de 0,5, sur 169 300 prêts de 2017 jamais vus
      pendant l'apprentissage</div>
  </div>
  <div class="tuile">
    <div class="chiffre-v">0,473</div>
    <div class="chiffre-l">de F1, le meilleur compromis des modèles testés entre défauts détectés et
      alertes justes (36 % au seuil de 0,5)</div>
  </div>
  <div class="tuile">
    <div class="chiffre-v">13 s</div>
    <div class="chiffre-l">pour réentraîner le modèle sur 1 045 257 prêts, contre 4 minutes pour
      un réseau de neurones</div>
  </div>
</div>

<div class="chiffre-l" style="margin:-0.4rem 0 1.2rem">En complément, indépendamment du seuil :
AUC de 0,7224, contre 0,681 pour la seule note attribuée par LendingClub.</div>

<div class="avis">Cet outil suggère une décision, il ne la prend pas : la décision finale appartient
au comité de crédit. Le modèle a été construit à des fins académiques sur des données publiques
LendingClub (États-Unis) ; son usage sur un portefeuille réel supposerait un réentraînement sur
les données de l'établissement.</div>
""", unsafe_allow_html=True)


# ===============================================================
# 10. ÉCRAN 1 — IDENTITÉ DU DEMANDEUR
# ===============================================================
elif st.session_state.page == "identite":
    st.markdown("""
<div class="titre-page">Qui fait la demande ?</div>
<div class="sous-titre">Ces informations apparaissent sur le rapport de synthèse.</div>
""", unsafe_allow_html=True)

    with carte("carte_identite"):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.text_input("Nom de famille", placeholder="Ex. Mbadinga", key="f_nom")
        with c2:
            st.text_input("Prénom", placeholder="Ex. Christelle", key="f_prenom")

        if st.session_state.erreur_identite:
            st.markdown('<div class="message erreur">Le nom et le prénom sont nécessaires pour continuer.</div>',
                        unsafe_allow_html=True)

    b1, _, b2 = st.columns([1, 1, 1])
    with b1:
        st.button("Retour à l'accueil", use_container_width=True,
                  on_click=aller, args=("accueil",), key="btn_retour_accueil")
    with b2:
        st.button("Continuer", type="primary", use_container_width=True,
                  on_click=valider_identite, key="btn_continuer")


# ===============================================================
# 11. ÉCRAN 2 — DOSSIER FINANCIER
#     Trois cartes empilées, trois champs par ligne : les champs
#     prennent toute la largeur disponible.
# ===============================================================
elif st.session_state.page == "dossier":
    nom_affiche = f"{st.session_state.f_prenom.strip()} {st.session_state.f_nom.strip()}".strip()
    st.markdown(f"""
<div class="titre-page">Le dossier de {nom_affiche}</div>
<div class="sous-titre">Renseignez les caractéristiques de la demande. Les valeurs proposées
correspondent à un dossier courant.</div>
""", unsafe_allow_html=True)

    with carte("carte_pret"):
        st.markdown('<div class="sec-titre">Le prêt</div><div class="sec-sous">Ce que le demandeur sollicite.</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.number_input("Montant demandé ($)", 500, 40000, step=500, key="f_loan_amnt")
        with c2:
            st.selectbox("Durée du prêt", list(DUREE_FR), format_func=DUREE_FR.get, key="f_term")
        with c3:
            st.number_input("Taux d'intérêt (%)", 5.0, 31.0, step=0.1, format="%.2f", key="f_int_rate")
        c4, c5 = st.columns([1, 2], gap="large")
        with c4:
            st.selectbox("Note LendingClub (A1 à G5)", SUB_GRADES, key="f_sub_grade")
        with c5:
            st.selectbox("Objet du financement", modalites("purpose"),
                         format_func=lambda p: PURPOSE_FR.get(p, p), key="f_purpose")

    with carte("carte_emprunteur"):
        st.markdown('<div class="sec-titre">L\'emprunteur</div>'
                    '<div class="sec-sous">Ses revenus, sa situation professionnelle et son logement.</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.number_input("Revenu annuel ($)", 0, 1_000_000, step=1000, key="f_annual_inc")
        with c2:
            st.selectbox("Vérification du revenu", list(VERIF),
                         format_func=lambda m: VERIF_FR.get(m, VERIF[m]), key="f_verif")
        with c3:
            st.number_input("Ratio d'endettement (%)", 0.0, 100.0, step=0.1, format="%.1f", key="f_dti")
        c4, c5, c6 = st.columns(3, gap="large")
        with c4:
            st.selectbox("Ancienneté dans l'emploi", ["Non renseignée"] + EMP_LENGTHS,
                         format_func=EMP_FR.get, key="f_emp_length")
        with c5:
            st.selectbox("Logement", modalites("home_ownership"),
                         format_func=lambda h: LOGEMENT_FR.get(h, h), key="f_home")
        with c6:
            st.selectbox("État de résidence", modalites("addr_state"),
                         format_func=lambda e: f"{ETATS_FR.get(e, e)} ({e})", key="f_etat")

    with carte("carte_historique"):
        st.markdown('<div class="sec-titre">L\'historique de crédit</div>'
                    '<div class="sec-sous">Ce que les bureaux de crédit savent de l\'emprunteur.</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.number_input("Score FICO (borne basse)", 600, 850, step=5, key="f_fico")
        with c2:
            st.date_input("Ouverture du premier crédit", min_value=dt.date(1940, 1, 1),
                          max_value=dt.date.today(), format="DD/MM/YYYY", key="f_premier_credit")
        with c3:
            st.number_input("Comptes ouverts sur 24 mois", 0, 60, step=1, key="f_acc_open")
        c4, c5, c6 = st.columns(3, gap="large")
        with c4:
            # Petit décalage pour aligner la case à cocher sur les champs voisins
            st.markdown('<div style="height:2.3rem"></div>', unsafe_allow_html=True)
            st.checkbox("Aucune demande récente", key="f_aucune_demande")
        aucune = st.session_state.f_aucune_demande
        with c5:
            st.number_input("Demandes de crédit sur 6 mois", 0, 40, step=1, disabled=aucune, key="f_inq_6m")
        with c6:
            st.number_input("Mois depuis la dernière demande", 0, 120, step=1, disabled=aucune, key="f_mths_inq")

    b1, _, b2 = st.columns([1, 1, 1])
    with b1:
        st.button("Retour", use_container_width=True,
                  on_click=aller, args=("identite",), key="btn_retour_identite")
    with b2:
        st.button("Lancer l'évaluation", type="primary", use_container_width=True,
                  on_click=aller, args=("resultat",), key="btn_evaluer")


# ===============================================================
# 12. ÉCRAN 3 — RÉSULTAT
# ===============================================================
elif st.session_state.page == "resultat":
    s = st.session_state
    prenom, nom = s.f_prenom.strip(), s.f_nom.strip()
    nom_complet = f"{prenom} {nom}".strip()

    client = dict(defauts)
    client.update({
        "loan_amnt": s.f_loan_amnt, "term": s.f_term, "int_rate": s.f_int_rate, "sub_grade": s.f_sub_grade,
        "annual_inc": s.f_annual_inc, "dti": s.f_dti, "fico_range_low": s.f_fico,
        "emp_length": None if s.f_emp_length == "Non renseignée" else s.f_emp_length,
        "acc_open_past_24mths": s.f_acc_open,
        "inq_last_6mths": 0 if s.f_aucune_demande else s.f_inq_6m,
        "mths_since_recent_inq": None if s.f_aucune_demande else s.f_mths_inq,
        "earliest_cr_line": s.f_premier_credit.isoformat(),
        "issue_d": dt.date.today().isoformat(), "purpose": s.f_purpose,
        "home_ownership": s.f_home, "verification_status": VERIF[s.f_verif], "addr_state": s.f_etat,
    })

    try:
        X = preparer_client(client, artefacts, colonnes)
    except ValueError as e:
        st.markdown(f'<div class="message erreur">Le dossier est incomplet : {e}</div>', unsafe_allow_html=True)
        st.button("Corriger le dossier", type="primary", on_click=aller, args=("dossier",), key="btn_corriger")
        st.stop()

    score = float(modele.predict_proba(X)[:, 1][0])
    refuse = score >= seuil

    valeurs = explainer.shap_values(X)
    if isinstance(valeurs, list):
        valeurs = valeurs[-1]
    valeurs = np.asarray(valeurs)
    if valeurs.ndim == 3:
        valeurs = valeurs[:, :, -1]
    base = float(np.ravel(explainer.expected_value)[-1])

    st.markdown(f"""
<div class="titre-page">Résultat pour {nom_complet}</div>
<div class="recap">Prêt de <b>{milliers(s.f_loan_amnt)} $</b> sur <b>{DUREE_FR[s.f_term]}</b>
à <b>{fr(s.f_int_rate)} %</b> · revenu annuel <b>{milliers(s.f_annual_inc)} $</b>
· ratio d'endettement <b>{fr(s.f_dti, 1)} %</b> · FICO <b>{s.f_fico}</b> · note <b>{s.f_sub_grade}</b></div>
<div style="height:1.2rem"></div>
""", unsafe_allow_html=True)

    gauche, droite = st.columns([1, 1.3], gap="large")

    with gauche:
        with carte("carte_verdict"):
            classe = "refus" if refuse else "accord"
            titre = "Dossier à risque" if refuse else "Dossier acceptable"
            detail = ("Le score atteint le seuil fixé par la politique de risque : un refus est suggéré."
                      if refuse else
                      "Le score reste sous le seuil fixé par la politique de risque.")
            couleur = "var(--rouge)" if refuse else "var(--vert)"
            position = min(max(seuil * 100, 0), 100)
            st.markdown(f"""
<div class="verdict {classe}">
  <div class="verdict-sur">Décision suggérée</div>
  <div class="verdict-titre">{titre}</div>
  <div class="verdict-detail">{detail}</div>
</div>
<div class="score"><span class="score-v">{fr(score, 3)}</span>
  <span class="score-l">score de risque, entre 0 et 1</span></div>
<div class="jauge">
  <div class="jauge-plein" style="width:{min(score, 1) * 100:.1f}%;background:{couleur}"></div>
  <div class="jauge-seuil" style="left:calc({position:.1f}% - 1.5px)"></div>
</div>
<div class="jauge-legende">
  <span style="left:clamp(0%, calc({position:.1f}% - 2.8rem), calc(100% - 5.6rem))">seuil {fr(seuil)}</span>
</div>
<div class="note">Le score classe les dossiers du moins risqué au plus risqué.
Il ne se lit pas comme un taux de défaut.</div>
""", unsafe_allow_html=True)

        with carte("carte_explication"):
            st.markdown(f"""
<div class="sec-titre">Ce qui a pesé dans la décision</div>
<div style="height:.5rem"></div>
<div class="explication">{expliquer_decision(valeurs[0], colonnes)}</div>
""", unsafe_allow_html=True)

        pdf = generer_pdf(score, refuse, seuil, client, valeurs[0], colonnes, nom_complet)
        st.download_button(
            "Télécharger la fiche de synthèse (PDF)", data=pdf,
            file_name=f"Synthese_credit_{nom_complet.replace(' ', '_') or 'client'}_{dt.date.today().isoformat()}.pdf",
            mime="application/pdf", use_container_width=True, key="btn_pdf",
        )
        b1, b2 = st.columns(2)
        with b1:
            st.button("Modifier le dossier", use_container_width=True,
                      on_click=aller, args=("dossier",), key="btn_modifier")
        with b2:
            st.button("Nouveau dossier", use_container_width=True,
                      on_click=nouvelle_evaluation, key="btn_nouvelle")

    with droite:
        with carte("carte_shap"):
            st.markdown('<div class="sec-titre">Le détail, variable par variable</div>'
                        '<div class="sec-sous">Les dix variables qui ont le plus déplacé le score.</div>',
                        unsafe_allow_html=True)
            plt.close("all")
            plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
            explication = shap.Explanation(values=valeurs[0], base_values=base, data=X.iloc[0].values,
                                           feature_names=[libelle(c) for c in colonnes])
            shap.plots.waterfall(explication, max_display=10, show=False)
            figure = plt.gcf()
            figure.set_facecolor("white")
            # SHAP écrit « 14 other features » en anglais : on traduit l'étiquette.
            for axe in figure.axes:
                etiquettes = [t.get_text().replace("other features", "autres variables")
                              for t in axe.get_yticklabels()]
                if any(etiquettes):
                    axe.set_yticks(axe.get_yticks())
                    axe.set_yticklabels(etiquettes)
            st.pyplot(figure, clear_figure=True)
            st.markdown('<div class="note">En rouge, les éléments qui rapprochent le dossier du défaut. '
                        'En bleu, ceux qui le rassurent. La longueur de chaque barre mesure le poids '
                        'de la variable pour ce dossier précis.</div>', unsafe_allow_html=True)


# ===============================================================
# 13. REMONTER EN HAUT DE PAGE À CHAQUE CHANGEMENT D'ÉCRAN
#     Streamlit garde la position de défilement : sans ceci, le
#     résultat s'ouvrirait au niveau du bouton « Lancer l'évaluation ».
#     Placé en dernier pour ne pas créer d'espace en haut de page.
# ===============================================================
if st.session_state.get("page_affichee") != st.session_state.page:
    st.session_state.page_affichee = st.session_state.page
    components.html("""
<script>
  const doc = window.parent.document;
  for (const sel of ['[data-testid="stMain"]', 'section.main', '[data-testid="stAppViewContainer"]']) {
    const zone = doc.querySelector(sel);
    if (zone) { zone.scrollTo({ top: 0 }); }
  }
  window.parent.scrollTo(0, 0);
</script>
""", height=0)
