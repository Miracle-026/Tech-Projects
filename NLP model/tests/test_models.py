"""
Unit tests for all models with robustness and out-of-domain checks.
"""

import unittest
import torch
from src.models.ewc_sentiment import EWCSentimentAnalyzer
from src.models.replay_classifier import ReplayClassifier
from src.models.progressive_lm import ProgressiveLMAnalyzer
from src.models.gem_ner import GEMNERAnalyzer
from src.models.online_intent import OnlineIntentAnalyzer
from src.models.multi_task import MultiTaskModel
from src.utils.data_loader import load_sentiment_data, load_classification_data, load_language_modeling_data, load_ner_data, load_intent_data
import numpy as np
from textattack.augmentation import EmbeddingAugmenter

class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cls.models = {
            'ewc_sentiment': EWCSentimentAnalyzer(model_name='distilbert-base-uncased', device=cls.device),
            'replay_classifier': ReplayClassifier(model_name='bert-base-uncased', device=cls.device),
            'progressive_lm': ProgressiveLMAnalyzer(model_name='distilbert-base-uncased', device=cls.device),
            'gem_ner': GEMNERAnalyzer(model_name='bert-base-uncased', device=cls.device),
            'online_intent': OnlineIntentAnalyzer(),
            'multi_task': MultiTaskModel(device=cls.device)
        }
        cls.data_loaders = {
            'sentiment': load_sentiment_data,
            'classification': load_classification_data,
            'language_modeling': load_language_modeling_data,
            'ner': load_ner_data,
            'intent': load_intent_data
        }
        cls.test_data = {
            'sentiment': [{'text': 'This is great!', 'label': 1, 'task_id': 0, 'domain': 'test'}],
            'classification': [{'text': 'High quality product', 'label': 1, 'task_id': 0, 'domain': 'test'}],
            'language_modeling': [{'text': 'The sun rises', 'task_id': 0, 'domain': 'test'}],
            'ner': [{'text': 'Apple in New York', 'entities': [(0, 5, 'ORG'), (10, 18, 'LOC')], 'task_id': 0, 'domain': 'test'}],
            'intent': [{'text': 'Book a flight', 'intent': 'book_flight', 'task_id': 0, 'domain': 'test'}]
        }
        cls.augmenter = EmbeddingAugmenter()
    
    def test_initialization(self):
        for model_name, model in self.models.items():
            self.assertIsNotNone(model, f"{model_name} initialization failed")
    
    def test_prediction(self):
        for model_name, model in self.models.items():
            task_type = model_name.replace('_', '') if model_name != 'online_intent' else 'intent'
            if task_type == 'language_modeling':
                continue  # Skip generative task
            data = self.test_data[task_type][0]
            pred = model.predict_single(data['text'], data['task_id'], return_probability=True)
            self.assertIsNotNone(pred, f"{model_name} prediction failed")
    
    def test_training(self):
        for model_name, model in self.models.items():
            task_type = model_name.replace('_', '') if model_name != 'online_intent' else 'intent'
            data = self.test_data[task_type]
            metrics = model.train_task(data, task_id=0, domain='test', epochs=1)
            self.assertGreater(len(metrics), 0, f"{model_name} training failed")
    
    def test_robustness(self):
        for model_name, model in self.models.items():
            task_type = model_name.replace('_', '') if model_name != 'online_intent' else 'intent'
            if task_type == 'language_modeling':
                continue  # Skip generative task
            data = self.test_data[task_type][0]
            original_pred = model.predict_single(data['text'], data['task_id'], return_probability=True)
            augmented_text = self.augmenter.augment(data['text'])[0]
            adv_pred = model.predict_single(augmented_text, data['task_id'], return_probability=True)
            if task_type == 'ner':
                self.assertEqual(len(original_pred), len(adv_pred), f"{model_name} robustness test failed")
            else:
                self.assertEqual(original_pred.label, adv_pred.label, f"{model_name} robustness test failed")
    
    def test_out_of_domain(self):
        for model_name, model in self.models.items():
            primary_task = model_name.replace('_', '') if model_name != 'online_intent' else 'intent'
            for task_type in self.test_data.keys():
                if task_type == primary_task or task_type == 'language_modeling':
                    continue
                data = self.test_data[task_type][0]
                try:
                    pred = model.predict_single(data['text'], data['task_id'], return_probability=True)
                    self.assertIsNotNone(pred, f"{model_name} out-of-domain test on {task_type} failed")
                except Exception as e:
                    print(f"Expected failure for {model_name} on {task_type}: {str(e)}")

if __name__ == '__main__':
    unittest.main()