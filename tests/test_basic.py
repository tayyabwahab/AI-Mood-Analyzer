"""
Basic tests for Audio Sentiment Analysis project.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.cnn_model import AudioSentimentCNN
from config.config import DATA_CONFIG, MODEL_CONFIG


class TestAudioSentimentCNN(unittest.TestCase):
    """Test cases for AudioSentimentCNN class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = AudioSentimentCNN(n_mfcc=30, nclass=16)
    
    def test_model_creation(self):
        """Test model creation."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.n_mfcc, 30)
        self.assertEqual(self.model.nclass, 16)
    
    def test_model_build(self):
        """Test model building."""
        model = self.model.build_model()
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 15)  # Expected number of layers
    
    def test_model_compile(self):
        """Test model compilation."""
        model = self.model.compile_model()
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
    
    def test_model_prediction_shape(self):
        """Test model prediction output shape."""
        self.model.build_model()
        self.model.compile_model()
        
        # Create dummy input
        dummy_input = np.random.randn(1, 30, 216, 1)
        
        # Make prediction
        prediction = self.model.predict(dummy_input)
        
        # Check output shape
        self.assertEqual(prediction.shape, (1, 16))


class TestConfig(unittest.TestCase):
    """Test cases for configuration."""
    
    def test_data_config(self):
        """Test data configuration."""
        self.assertIn('sampling_rate', DATA_CONFIG)
        self.assertIn('audio_duration', DATA_CONFIG)
        self.assertIn('n_mfcc', DATA_CONFIG)
        self.assertEqual(DATA_CONFIG['sampling_rate'], 44100)
    
    def test_model_config(self):
        """Test model configuration."""
        self.assertIn('nclass', MODEL_CONFIG)
        self.assertIn('learning_rate', MODEL_CONFIG)
        self.assertEqual(MODEL_CONFIG['nclass'], 16)


if __name__ == '__main__':
    unittest.main()
