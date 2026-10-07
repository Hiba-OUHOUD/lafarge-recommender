
---

###  README pour `lafarge-recommender`

```markdown
#  Lafarge E-Learning - Microservice ML

> Microservice de recommandation basé sur l'algorithme **SVD (Singular Value Decomposition)** pour personnaliser les formations proposées à chaque employé.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  Objectif

Ce microservice implémente un **système de recommandation intelligent** qui apprend du comportement des utilisateurs pour leur proposer les formations les plus pertinentes.

Contrairement à un simple filtrage par domaine, le modèle **s'améliore continuellement** en fonction des interactions utilisateurs (inscriptions, progression, quiz, clics).

---

##  Principe de l'algorithme SVD

1. **Collecte des interactions** : inscriptions, progression, quiz réussis, clics d'intérêt
2. **Construction de la matrice** : matrice creuse **utilisateurs × formations**
3. **Décomposition SVD** : réduction en **facteurs latents**
4. **Prédiction** : calcul des scores pour toutes les formations
5. **Recommandation** : tri par score décroissant

**Formule du score d'interaction :**
- Inscription : `+1`
- Progression : `+progress/20` (jusqu'à `+5`)
- Quiz réussi : `+10`
- Clic sur un domaine : `+click_count`

---

##  Technologies utilisées

| Catégorie | Technologie |
|-----------|-------------|
| **Langage** | Python 3.13 |
| **Framework API** | Flask 3.0 |
| **Machine Learning** | Scikit-learn (TruncatedSVD) |
| **Data** | Pandas + NumPy |
| **BDD** | MySQL (mysql-connector-python) |
| **Serveur** | Gunicorn (production) |

---

##  Installation

### Prérequis
- Python 3.10+
- MySQL (XAMPP ou serveur)

### Étapes

```bash
# Cloner le repository
git clone https://github.com/Hiba-OUHOUD/lafarge-recommender.git

# Entrer dans le dossier
cd lafarge-recommender

# Créer un environnement virtuel (optionnel mais recommandé)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt

# Lancer le microservice
python app.py
