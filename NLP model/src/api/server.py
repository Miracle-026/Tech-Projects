"""
Flask API server with WebSocket support for model inference.
"""

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import logging
import re
from pathlib import Path
from src.models.ewc_sentiment import EWCSentimentAnalyzer
from src.models.replay_classifier import ReplayClassifier
from src.models.progressive_lm import ProgressiveLMAnalyzer
from src.models.gem_ner import GEMNERAnalyzer
from src.models.online_intent import OnlineIntentAnalyzer
from src.models.multi_task import MultiTaskModel
from src.models.ensemble import EnsembleModel
from flask_httpauth import HTTPTokenAuth

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
auth = HTTPTokenAuth(scheme='Bearer')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("APIServer")

# Authentication
@auth.verify_token
def verify_token(token):
    valid_tokens = {'your-secret-api-key': 'user'}  # Store securely
    return token in valid_tokens

# Load models
models = {
    'ewc_sentiment': None,
    'replay_classifier': None,
    'progressive_lm': None,
    'gem_ner': None,
    'online_intent': None,
    'multi_task': None,
    'ensemble': None
}

def validate_input(data):
    """Validate input data."""
    if not isinstance(data, dict) or 'text' not in data or 'task_id' not in data:
        return False, 'Missing text or task_id'
    if not isinstance(data['text'], str) or not isinstance(data['task_id'], int):
        return False, 'Invalid text or task_id type'
    if len(data['text']) > 1000:
        return False, 'Text too long'
    if re.search(r'<\s*script|[\'";]', data['text'], re.IGNORECASE):
        return False, 'Invalid characters in text'
    return True, None

def load_models(checkpoint_dir: str = 'results/models/'):
    """Load all models from checkpoints."""
    checkpoint_dir = Path(checkpoint_dir)
    model_classes = {
        'ewc_sentiment': EWCSentimentAnalyzer,
        'replay_classifier': ReplayClassifier,
        'progressive_lm': ProgressiveLMAnalyzer,
        'gem_ner': GEMNERAnalyzer,
        'online_intent': OnlineIntentAnalyzer,
        'multi_task': MultiTaskModel
    }
    
    for model_name, model_class in model_classes.items():
        ext = 'pt' if model_name in ['ewc_sentiment', 'progressive_lm', 'gem_ner', 'multi_task'] else 'ckpt' if model_name == 'replay_classifier' else 'pkl'
        checkpoint_path = list(checkpoint_dir.glob(f"{model_name}_*_task_*.{ext}"))
        if checkpoint_path:
            models[model_name] = model_class.load_from_checkpoint(str(checkpoint_path[0]))
            logger.info(f"Loaded {model_name} from {checkpoint_path[0]}")
    
    individual_models = [m for m in models.values() if m is not None]
    if individual_models:
        models['ensemble'] = EnsembleModel(models=individual_models, save_dir=str(checkpoint_dir))
        logger.info("Loaded ensemble model")

@app.route('/health', methods=['GET'])
@auth.login_required
def health():
    """Health check endpoint."""
    loaded_models = [name for name, model in models.items() if model is not None]
    return jsonify({'status': 'healthy', 'models': loaded_models})

@app.route('/predict/<model_name>', methods=['POST'])
@auth.login_required
def predict(model_name: str):
    """Single prediction endpoint."""
    if model_name not in models or models[model_name] is None:
        return jsonify({'error': f"Model {model_name} not loaded"}), 400
    
    data = request.get_json()
    valid, error = validate_input(data)
    if not valid:
        return jsonify({'error': error}), 400
    
    try:
        task_type = model_name.replace('_', '') if model_name != 'online_intent' else 'intent'
        if model_name == 'ensemble':
            prediction = models[model_name].predict_single(data['text'], data['task_id'], task_type, return_probability=True)
        else:
            prediction = models[model_name].predict_single(
                data['text'], data['task_id'], task_type=task_type, return_probability=True if task_type in ['sentiment', 'classification', 'intent'] else False
            )
        
        response = {'text': data['text'], 'task_id': data['task_id']}
        if task_type in ['sentiment', 'classification']:
            response.update({'label': prediction.label, 'probability': prediction.probability})
        elif task_type == 'intent':
            response.update({'intent': prediction.intent, 'probability': prediction.probability})
        elif task_type == 'ner':
            response.update({'entities': prediction})
        elif task_type == 'language_modeling':
            response.update({'output': prediction})
        return jsonify(response)
    except Exception as e:
        logger.error(f"Prediction error for {model_name}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict/<model_name>', methods=['POST'])
@auth.login_required
def batch_predict(model_name: str):
    """Batch prediction endpoint."""
    if model_name not in models or models[model_name] is None:
        return jsonify({'error': f"Model {model_name} not loaded"}), 400
    
    data = request.get_json()
    if not data or not isinstance(data.get('inputs'), list):
        return jsonify({'error': 'Inputs must be a list of {text, task_id} objects'}), 400
    
    try:
        task_type = model_name.replace('_', '') if model_name != 'online_intent' else 'intent'
        predictions = []
        for item in data['inputs']:
            valid, error = validate_input(item)
            if not valid:
                predictions.append({'error': error})
                continue
            if model_name == 'ensemble':
                pred = models[model_name].predict_single(item['text'], item['task_id'], task_type, return_probability=True)
            else:
                pred = models[model_name].predict_single(
                    item['text'], item['task_id'], task_type=task_type, return_probability=True if task_type in ['sentiment', 'classification', 'intent'] else False
                )
            
            response = {'text': item['text'], 'task_id': item['task_id']}
            if task_type in ['sentiment', 'classification']:
                response.update({'label': pred.label, 'probability': pred.probability})
            elif task_type == 'intent':
                response.update({'intent': pred.intent, 'probability': pred.probability})
            elif task_type == 'ner':
                response.update({'entities': pred})
            elif task_type == 'language_modeling':
                response.update({'output': pred})
            predictions.append(response)
        
        return jsonify({'predictions': predictions})
    except Exception as e:
        logger.error(f"Batch prediction error for {model_name}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@socketio.on('predict')
def handle_websocket_predict(data):
    """WebSocket prediction endpoint."""
    if not data or 'model_name' not in data:
        emit('error', {'error': 'Missing model_name'})
        return
    
    model_name = data['model_name']
    if model_name not in models or models[model_name] is None:
        emit('error', {'error': f"Model {model_name} not loaded"})
        return
    
    valid, error = validate_input(data)
    if not valid:
        emit('error', {'error': error})
        return
    
    try:
        task_type = model_name.replace('_', '') if model_name != 'online_intent' else 'intent'
        if model_name == 'ensemble':
            prediction = models[model_name].predict_single(data['text'], data['task_id'], task_type, return_probability=True)
        else:
            prediction = models[model_name].predict_single(
                data['text'], data['task_id'], task_type=task_type, return_probability=True if task_type in ['sentiment', 'classification', 'intent'] else False
            )
        
        response = {'text': data['text'], 'task_id': data['task_id']}
        if task_type in ['sentiment', 'classification']:
            response.update({'label': prediction.label, 'probability': prediction.probability})
        elif task_type == 'intent':
            response.update({'intent': prediction.intent, 'probability': prediction.probability})
        elif task_type == 'ner':
            response.update({'entities': prediction})
        elif task_type == 'language_modeling':
            response.update({'output': prediction})
        emit('prediction', response)
    except Exception as e:
        logger.error(f"WebSocket prediction error for {model_name}: {str(e)}")
        emit('error', {'error': str(e)})

if __name__ == '__main__':
    load_models()
    socketio.run(app, host='0.0.0.0', port=5000)