"""
Utilities for model explainability using SHAP and token-level importance.
"""

import logging
from typing import Dict, List, Any
import torch
import numpy as np
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

def setup_logging(verbose: bool) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("Explainability")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def explain_classification(model, text: str, task_id: int, model_name: str = 'bert-base-uncased', device: str = 'cpu') -> Dict[str, Any]:
    """Generate SHAP explanations for classification models."""
    logger = setup_logging(model.verbose)
    explainer_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    explainer_model.load_state_dict(model.bert.state_dict(), strict=False)  # Load compatible weights
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Define prediction function
    def predict(texts):
        inputs = tokenizer(texts, return_tensors='pt', max_length=128, padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = explainer_model(**inputs).logits
        return torch.softmax(outputs, dim=-1).cpu().numpy()
    
    # Initialize SHAP explainer
    explainer = shap.KernelExplainer(predict, [text])
    shap_values = explainer.shap_values([text])[0]
    
    # Format results
    tokens = tokenizer.tokenize(text)
    return {
        'text': text,
        'task_id': task_id,
        'tokens': tokens,
        'shap_values': shap_values.tolist(),
        'base_value': explainer.expected_value.tolist()
    }

def explain_ner(model, text: str, task_id: int, model_name: str = 'bert-base-uncased', device: str = 'cpu') -> Dict[str, Any]:
    """Generate token-level importance for NER models."""
    logger = setup_logging(model.verbose)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors='pt', max_length=128, padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.forward(inputs)
        scores = torch.softmax(outputs, dim=-1).squeeze(0).cpu().numpy()
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
    token_importance = [
        {'token': token, 'importance': scores[i].max(), 'label': np.argmax(scores[i])}
        for i, token in enumerate(tokens) if token not in ['[CLS]', '[SEP]', '[PAD]']
    ]
    
    return {
        'text': text,
        'task_id': task_id,
        'token_importance': token_importance
    }

def explain_prediction(model, text: str, task_id: int, task_type: str, model_name: str = 'bert-base-uncased', device: str = 'cpu') -> Dict[str, Any]:
    """Generate explanation based on task type."""
    if task_type in ['sentiment', 'classification', 'intent']:
        return explain_classification(model, text, task_id, model_name, device)
    elif task_type == 'ner':
        return explain_ner(model, text, task_id, model_name, device)
    else:
        return {'error': 'Explainability not supported for language modeling'}