"""
Microservice de recommandation ML
Expose une API REST qui prédit les formations les plus pertinentes pour un utilisateur.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import ml_model
import threading
import time

app = Flask(__name__)
CORS(app)

# Modèle global (rechargé périodiquement)
MODEL = None
MODEL_LOCK = threading.Lock()


def train_model():
    """Entraîne le modèle et le stocke en mémoire."""
    global MODEL
    print("🔄 Entraînement du modèle en arrière-plan...")
    new_model = ml_model.build_model()
    with MODEL_LOCK:
        MODEL = new_model
    print("✅ Modèle mis à jour")


def background_trainer():
    """Ré-entraîne le modèle toutes les 5 minutes."""
    while True:
        time.sleep(300)  # 5 minutes
        train_model()


@app.route('/health', methods=['GET'])
def health():
    """Vérifie que le service est actif."""
    with MODEL_LOCK:
        return jsonify({
            'status': 'ok',
            'model_trained': MODEL is not None
        })


@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend_endpoint(user_id):
    """Retourne les formations recommandées pour un utilisateur."""
    top_n = int(request.args.get('top', 10))

    with MODEL_LOCK:
        model = MODEL

    if model is None:
        return jsonify({'error': 'Modèle non entraîné', 'recommendations': []}), 503

    recommendations = ml_model.recommend(user_id, model, top_n=top_n)

    return jsonify({
        'userId': user_id,
        'count': len(recommendations),
        'recommendations': recommendations
    })


@app.route('/train', methods=['POST'])
def force_train():
    """Force un ré-entraînement immédiat du modèle."""
    train_model()
    return jsonify({'message': 'Modèle ré-entraîné'})


if __name__ == '__main__':
    # Entraîner le modèle au démarrage
    train_model()

    # Lancer l'entraînement périodique en arrière-plan
    trainer_thread = threading.Thread(target=background_trainer, daemon=True)
    trainer_thread.start()

    print("🚀 Microservice de recommandation démarré sur http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)