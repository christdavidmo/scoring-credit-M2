# Scoring adaptatif et explicable du risque de crédit

Outil d'aide à la décision qui estime le risque de défaut d'une demande de prêt, explique chaque décision (SHAP) et s'inscrit dans un cadre de gestion adaptative : surveillance de la dérive des données (PSI) et ré-entraînement du modèle.

**Application en ligne :** https://scoring-credit-mondjot-cd.streamlit.app

> Projet réalisé dans le cadre du mémoire de Master 2 (Digital Campus, Groupe ISM, Dakar) : *Cadre de gestion adaptative du risque de crédit fondé sur l'intelligence artificielle pour les institutions financières*.

---

## Ce que fait l'outil

1. **Saisie** de l'identité du demandeur et de son dossier (prêt, emprunteur, historique de crédit).
2. **Score de risque** calculé par un modèle LightGBM.
3. **Décision suggérée** : le score est comparé à un seuil de refus réglable.
4. **Explication** : les facteurs qui ont pesé dans la décision, en phrase et en graphique (SHAP).
5. **Rapport PDF** téléchargeable, à joindre au dossier.
6. **Réglage du seuil** selon la politique de risque, avec son effet mesuré.

L'outil **suggère** une décision : la décision finale reste humaine.

---

## Données

- Jeu public **LendingClub** (prêts émis entre 2007 et 2018), utilisé faute d'accès aux données d'une banque de la zone CEMAC.
- Prêts dont l'issue est connue, variables **disponibles au moment de l'octroi** uniquement (pas de fuite d'information).
- **Découpage temporel** : apprentissage sur octobre 2012 à décembre 2016 (1 045 257 prêts), test sur 2017 (169 300 prêts). L'année 2018 sert à mesurer la dérive.

Les données ne sont pas incluses dans ce dépôt.

---

## Méthode

| Étape | Contenu |
|---|---|
| Sélection du modèle | Validation croisée **temporelle** sur 2012-2016 : trois plis (test 2014, 2015, 2016), le test étant toujours postérieur à l'apprentissage |
| Modèles comparés | Régression logistique, Random Forest, XGBoost, LightGBM, réseau de neurones (MLP, PyTorch) et la note LendingClub comme référence |
| Réglage | Grille de 12 combinaisons pour LightGBM, jugée sur 2016 |
| Évaluation finale | Une seule fois, sur 2017 |
| Explicabilité | SHAP (TreeExplainer) : contribution de chaque variable, pour chaque dossier |
| Adaptativité | PSI sur 2018, puis comparaison d'un modèle ré-entraîné avec l'ancien |
| Suivi | Expériences et versions des modèles enregistrées dans **MLflow** |

---

## Résultats

**Validation croisée temporelle (moyenne sur 2014, 2015, 2016, réglages de départ)**

| Modèle | AUC | F1 | Rappel | Précision | Temps (3 plis) |
|---|---|---|---|---|---|
| **LightGBM** | **0,726** | **0,447** | 0,668 | 0,337 | 34 s |
| XGBoost | 0,725 | 0,446 | 0,659 | 0,339 | 38 s |
| MLP | 0,724 | 0,445 | 0,683 | 0,332 | 358 s |
| Random Forest | 0,722 | 0,444 | 0,600 | 0,353 | 179 s |
| Régression logistique | 0,721 | 0,444 | 0,637 | 0,342 | 28 s |
| Note LendingClub | 0,697 | | | | |

LightGBM obtient la meilleure AUC et le meilleur F1, se classe premier ou à égalité chaque année, et s'entraîne dix fois plus vite que le réseau de neurones. Les écarts avec XGBoost et le MLP restent faibles : le choix est confirmé par le temps de calcul, le faible surapprentissage et l'explicabilité exacte.

**Modèle retenu : LightGBM réglé, évalué sur 2017**

| AUC | Rappel | Précision | F1 |
|---|---|---|---|
| 0,722 | 0,689 | 0,360 | 0,473 |

**Effet du seuil de refus (prêts de 2017)**

| Seuil | Défauts détectés | Alertes justes |
|---|---|---|
| 0,30 | 93 % | 28 % |
| 0,50 | 69 % | 36 % |
| 0,70 | 27 % | 49 % |

**Dérive et ré-entraînement.** Le PSI détecte une dérive nette sur 2018 (`revol_util` : 0,338, au-delà du seuil de 0,25). Un modèle ré-entraîné sur 2012-2017 dépasse l'ancien de +0,013 point d'AUC sur 2018 (IC 95 % : +0,012 à +0,014). Ce résultat est indicatif : les étiquettes de 2018 sont incomplètes (prêts trop récents).

---

## Structure du dépôt

```
├── src/
│   ├── app.py            interface Streamlit
│   ├── api.py            API d'inférence (FastAPI)
│   └── preparation.py    préparation d'un dossier (mêmes règles qu'à l'entraînement)
├── models/
│   ├── lightgbm_final.joblib
│   ├── preprocessing_artefacts.json
│   └── valeurs_par_defaut.json
├── .streamlit/config.toml
└── requirements.txt
```

---

## Lancer en local (Python 3.12)

```bash
pip install -r requirements.txt

# Interface
python -m streamlit run src/app.py

# API (depuis le dossier src)
cd src
python -m uvicorn api:app --port 8000
```

Documentation interactive de l'API : http://127.0.0.1:8000/docs

| Route | Rôle |
|---|---|
| `GET /sante` | Vérifie que l'API fonctionne |
| `GET /modele` | Fiche d'identité du modèle (période, réglages, performances) |
| `GET /modalites` | Valeurs acceptées pour les variables catégorielles |
| `POST /predire` | Score, décision suggérée et 5 principaux facteurs SHAP |

---

## Limites

- Prototype de recherche, **pas un système de production bancaire** : ni authentification, ni base de données, ni surveillance automatique.
- Données américaines : le modèle n'est pas transposable tel quel à la clientèle de la zone CEMAC.
- Le score n'est **pas une probabilité calibrée** (pondération des classes à l'entraînement) : il se lit comme un classement du risque.

---

## Auteur

**Christ David Mondjot**, Master 2 Big Data et Intelligence Artificielle, Digital Campus (Groupe ISM), Dakar
Encadrement : M. Momar
