# Security Guide

This document outlines security practices for the Flask API (`src/api/server.py`) in the `nlp-continual-learning` repository to ensure secure operation in production.

## 1. Input Validation

### Practices
- **Validate JSON Inputs**: Ensure all API requests (`/predict`, `/batch_predict`) contain required fields (`text`, `task_id`) and correct data types.
- **Sanitize Text Inputs**: Prevent injection attacks by sanitizing text inputs to remove malicious code or special characters.
- **Limit Input Size**: Restrict input text length to prevent denial-of-service (DoS) attacks.

### Implementation
```python
from flask import request, jsonify
import re

def validate_input(data):
    if not isinstance(data, dict) or 'text' not in data or 'task_id' not in data:
        return False, 'Missing text or task_id'
    if not isinstance(data['text'], str) or not isinstance(data['task_id'], int):
        return False, 'Invalid text or task_id type'
    if len(data['text']) > 1000:  # Limit text length
        return False, 'Text too long'
    # Remove potential script tags or SQL injection patterns
    if re.search(r'<\s*script|[\'";]', data['text'], re.IGNORECASE):
        return False, 'Invalid characters in text'
    return True, None

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name: str):
    data = request.get_json()
    valid, error = validate_input(data)
    if not valid:
        return jsonify({'error': error}), 400
    # Proceed with prediction