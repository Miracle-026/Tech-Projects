"""
Unit tests for utility modules.
"""

import unittest
import tempfile
import os
from pathlib import Path
import json
from src.utils.data_loader import load_ner_data, load_intent_data, IntentExample, NERExample
from src.utils.visualization import plot_training_metrics, plot_forgetting_scores, plot_domain_summary
from src.core.base_learner import TaskMetrics

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def test_load_ner_data(self):
        # Create sample NER data
        sample_data = [
            {"text": "Apple Inc. in New York", "entities": [[0, 9, "ORG"], [14, 22, "LOC"]]}
        ]
        sample_path = os.path.join(self.temp_dir, 'test_ner.json')
        with open(sample_path, 'w') as f:
            json.dump(sample_data, f)
        
        examples = load_ner_data(sample_path, task_id=0, domain="news")
        self.assertEqual(len(examples), 1)
        self.assertIsInstance(examples[0], NERExample)
        self.assertEqual(examples[0].text, "Apple Inc. in New York")
        self.assertEqual(examples[0].entities, [(0, 9, "ORG"), (14, 22, "LOC")])
    
    def test_load_intent_data(self):
        # Create sample intent data
        sample_data = [
            {"text": "Book a flight", "intent": "book_flight"}
        ]
        sample_path = os.path.join(self.temp_dir, 'test_intent.json')
        with open(sample_path, 'w') as f:
            json.dump(sample_data, f)
        
        examples = load_intent_data(sample_path, task_id=0, domain="travel")
        self.assertEqual(len(examples), 1)
        self.assertIsInstance(examples[0], IntentExample)
        self.assertEqual(examples[0].text, "Book a flight")
        self.assertEqual(examples[0].intent, "book_flight")
    
    def test_plot_training_metrics(self):
        metrics_history = {
            "news": [
                TaskMetrics(accuracy=0.8, loss=0.2, precision=0.8, recall=0.8, f1_score=0.8, task_id=0, epoch=0),
                TaskMetrics(accuracy=0.85, loss=0.15, precision=0.85, recall=0.85, f1_score=0.85, task_id=0, epoch=1)
            ]
        }
        output_path = os.path.join(self.temp_dir, 'metrics.png')
        plot_training_metrics(metrics_history, output_path, metric_types=['accuracy', 'f1_score'])
        self.assertTrue(os.path.exists(output_path))
    
    def test_plot_forgetting_scores(self):
        forgetting_scores = {"news": 0.05, "medical": 0.03}
        output_path = os.path.join(self.temp_dir, 'forgetting.png')
        plot_forgetting_scores(forgetting_scores, output_path)
        self.assertTrue(os.path.exists(output_path))
    
    def test_plot_domain_summary(self):
        summary = {
            "domain_performance": {
                "news": {"best_accuracy": 0.9, "best_f1": 0.88, "task_id": 0, "labels": ["ORG", "LOC"]},
                "medical": {"best_accuracy": 0.87, "best_f1": 0.86, "task_id": 1, "labels": ["DISEASE"]}
            }
        }
        output_path = os.path.join(self.temp_dir, 'summary.png')
        plot_domain_summary(summary, output_path, metric="best_accuracy")
        self.assertTrue(os.path.exists(output_path))
    
    def tearDown(self):
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

if __name__ == '__main__':
    unittest.main()