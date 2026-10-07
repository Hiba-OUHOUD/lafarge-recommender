"""
Modèle de recommandation ML
Algorithme : SVD (Singular Value Decomposition) via TruncatedSVD de sklearn
Principe : Factorisation matricielle de la matrice utilisateur × formation
"""

import mysql.connector
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

# Configuration de la connexion MySQL
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '',
    'database': 'lafargedb'
}


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def load_interactions():
    """
    Charge les interactions utilisateur-formation depuis la base.
    Score d'interaction (implicite) :
    - Inscription (enrollment) : +1
    - Progression : +progress/20 (jusqu'à +5)
    - Quiz réussi : +10
    - Clic d'intérêt : +1 par clic
    """
    conn = get_connection()

    # Score de base : inscription + progression
    query = """
        SELECT 
            e.employee_id AS user_id,
            e.training_id AS training_id,
            1 + (e.progress / 20.0) AS base_score
        FROM enrollments e
    """
    df = pd.read_sql(query, conn)

    # Bonus : quiz réussis
    query2 = """
        SELECT 
            e.employee_id AS user_id,
            e.training_id AS training_id,
            10 AS base_score
        FROM results r
        JOIN enrollments e ON e.id = r.enrollment_id
        WHERE r.passed = 1
    """
    try:
        df2 = pd.read_sql(query2, conn)
        df = pd.concat([df, df2], ignore_index=True)
    except Exception as e:
        print(f"⚠️ Impossible de charger results : {e}")

    # Bonus : clics d'intérêt (par domaine)
    query3 = """
        SELECT 
            ui.user_id,
            t.id AS training_id,
            ui.click_count AS base_score
        FROM user_interests ui
        JOIN trainings t ON t.domain = ui.domain
    """
    try:
        df3 = pd.read_sql(query3, conn)
        df = pd.concat([df, df3], ignore_index=True)
    except Exception as e:
        print(f"⚠️ Impossible de charger user_interests (table peut-être absente) : {e}")

    conn.close()

    # Agréger par (user_id, training_id)
    df = df.groupby(['user_id', 'training_id'], as_index=False)['base_score'].sum()
    df['base_score'] = pd.to_numeric(df['base_score'], errors='coerce').fillna(0).astype(float)
    return df


def build_model():
    """Construit et entraîne le modèle de factorisation matricielle."""
    print("📊 Chargement des interactions...")
    interactions = load_interactions()
    print(f"✅ {len(interactions)} interactions chargées")

    if len(interactions) < 3:
        print("⚠️ Pas assez de données pour entraîner un modèle ML.")
        return None

    # Encoder les IDs en indices continus
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    interactions['user_idx'] = user_encoder.fit_transform(interactions['user_id'])
    interactions['item_idx'] = item_encoder.fit_transform(interactions['training_id'])

    n_users = interactions['user_idx'].nunique()
    n_items = interactions['item_idx'].nunique()

    print(f"👥 {n_users} utilisateurs × 📚 {n_items} formations")

    # ⚠️ CORRECTION : Forcer le type float64 pour scipy
    interactions['base_score'] = pd.to_numeric(interactions['base_score'], errors='coerce').fillna(0).astype(float)

    print(f"🔢 Type de base_score : {interactions['base_score'].dtype}")

    # Créer la matrice creuse user × item
    matrix = csr_matrix(
        (interactions['base_score'],
         (interactions['user_idx'], interactions['item_idx'])),
        shape=(n_users, n_items)
    )

    # SVD : décomposition en facteurs latents
    n_components = min(10, min(n_users, n_items) - 1)
    if n_components < 1:
        n_components = 1

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(matrix)
    item_factors = svd.components_.T

    print(f"🧠 Modèle entraîné : variance expliquée = {svd.explained_variance_ratio_.sum():.2%}")

    return {
        'svd': svd,
        'user_factors': user_factors,
        'item_factors': item_factors,
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'n_users': n_users,
        'n_items': n_items,
    }


def recommend(user_id, model, top_n=10):
    """Prédit les top_n formations pour un utilisateur."""
    if model is None:
        return []

    try:
        user_idx = model['user_encoder'].transform([user_id])[0]
    except ValueError:
        print(f"⚠️ Utilisateur {user_id} inconnu du modèle")
        return []

    scores = np.dot(model['user_factors'][user_idx], model['item_factors'].T)
    top_indices = np.argsort(scores)[::-1]

    results = []
    for idx in top_indices[:top_n]:
        training_id = int(model['item_encoder'].inverse_transform([idx])[0])
        results.append({
            'trainingId': training_id,
            'score': float(scores[idx])
        })

    return results


if __name__ == '__main__':
    print("🚀 Entraînement du modèle...")
    model = build_model()
    if model:
        print("\n🎯 Recommandations pour l'utilisateur 8 (Ahmed) :")
        recs = recommend(8, model, top_n=5)
        for r in recs:
            print(f"   Formation #{r['trainingId']} - Score : {r['score']:.2f}")