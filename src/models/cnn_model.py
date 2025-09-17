"""
CNN model architecture for audio sentiment analysis.
Implements 2D CNN for MFCC feature classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses
from config.config import MODEL_CONFIG


class AudioSentimentCNN:
    """2D CNN model for audio sentiment analysis using MFCC features."""
    
    def __init__(self, n_mfcc=30, nclass=16):
        """
        Initialize the CNN model.
        
        Args:
            n_mfcc (int): Number of MFCC coefficients
            nclass (int): Number of output classes
        """
        self.n_mfcc = n_mfcc
        self.nclass = nclass
        self.model = None
        
    def build_model(self):
        """
        Build the 2D CNN model architecture.
        
        Returns:
            tf.keras.Model: Compiled CNN model
        """
        # Input layer
        inp = layers.Input(shape=(self.n_mfcc, 216, 1))
        
        # First convolutional block
        x = layers.Conv2D(32, (4, 10), padding="same")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Dropout(rate=MODEL_CONFIG['dropout_rate'])(x)
        
        # Second convolutional block
        x = layers.Conv2D(32, (4, 10), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Dropout(rate=MODEL_CONFIG['dropout_rate'])(x)
        
        # Third convolutional block
        x = layers.Conv2D(32, (4, 10), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Dropout(rate=MODEL_CONFIG['dropout_rate'])(x)
        
        # Fourth convolutional block
        x = layers.Conv2D(32, (4, 10), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPool2D()(x)
        x = layers.Dropout(rate=MODEL_CONFIG['dropout_rate'])(x)
        
        # Dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(64)(x)
        x = layers.Dropout(rate=MODEL_CONFIG['dropout_rate'])(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(rate=MODEL_CONFIG['dropout_rate'])(x)
        
        # Output layer
        out = layers.Dense(self.nclass, activation="softmax")(x)
        
        # Create model
        self.model = models.Model(inputs=inp, outputs=out)
        
        return self.model
    
    def compile_model(self):
        """
        Compile the model with optimizer, loss, and metrics.
        
        Returns:
            tf.keras.Model: Compiled model
        """
        if self.model is None:
            self.build_model()
        
        # Configure optimizer
        if MODEL_CONFIG['optimizer'].lower() == 'adam':
            opt = optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate'])
        else:
            opt = optimizers.RMSprop(learning_rate=MODEL_CONFIG['learning_rate'])
        
        # Compile model
        self.model.compile(
            optimizer=opt,
            loss=MODEL_CONFIG['loss'],
            metrics=MODEL_CONFIG['metrics']
        )
        
        return self.model
    
    def get_model_summary(self):
        """
        Get model architecture summary.
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            self.build_model()
        
        return self.model.summary()
    
    def save_model(self, filepath):
        """
        Save the model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def save_model_json(self, filepath):
        """
        Save model architecture as JSON.
        
        Args:
            filepath (str): Path to save the JSON file
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        model_json = self.model.to_json()
        with open(filepath, "w") as json_file:
            json_file.write(model_json)
        print(f"Model JSON saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a saved model.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
    
    def load_model_from_json(self, json_path, weights_path):
        """
        Load model from JSON and weights files.
        
        Args:
            json_path (str): Path to the JSON file
            weights_path (str): Path to the weights file
        """
        with open(json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        
        self.model = tf.keras.models.model_from_json(loaded_model_json)
        self.model.load_weights(weights_path)
        
        # Compile the model
        self.compile_model()
        
        print(f"Model loaded from {json_path} and {weights_path}")
        return self.model
