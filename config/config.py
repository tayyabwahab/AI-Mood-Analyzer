"""
Configuration file for Audio Sentiment Analysis project.
Contains all hyperparameters, paths, and model settings.
"""

import os

# Data Configuration
DATA_CONFIG = {
    'sampling_rate': 44100,
    'audio_duration': 2.5,
    'n_mfcc': 30,
    'n_melspec': 128,
    'test_size': 0.25,
    'random_state': 42,
    'batch_size': 16,
    'epochs': 50
}

# Model Configuration
MODEL_CONFIG = {
    'nclass': 16,
    'learning_rate': 0.001,
    'dropout_rate': 0.2,
    'optimizer': 'adam',
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy']
}

# Paths Configuration
PATHS = {
    'data_dir': 'data',
    'models_dir': 'saved_models',
    'output_dir': 'output',
    'logs_dir': 'logs'
}

# Emotion labels mapping
EMOTION_MAPPING = {
    1: 'neutral',
    2: 'calm', 
    3: 'happy',
    4: 'sad',
    5: 'angry',
    6: 'fear',
    7: 'disgust',
    8: 'surprise'
}

# Gender mapping
GENDER_MAPPING = {
    'even': 'female',
    'odd': 'male'
}

# Create necessary directories
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)
