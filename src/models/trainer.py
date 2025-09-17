"""
Training module for audio sentiment analysis model.
Handles model training, validation, and evaluation.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    TensorBoard
)
from config.config import DATA_CONFIG, PATHS
from .cnn_model import AudioSentimentCNN


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, model=None):
        """
        Initialize the trainer.
        
        Args:
            model: Pre-built model (optional)
        """
        self.model = model
        self.label_encoder = LabelEncoder()
        self.history = None
        
    def prepare_data(self, X, y):
        """
        Prepare data for training by splitting and encoding labels.
        
        Args:
            X (np.ndarray): Feature data
            y (np.ndarray): Target labels
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=DATA_CONFIG['test_size'], 
            shuffle=True, 
            random_state=DATA_CONFIG['random_state']
        )
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Convert to categorical
        y_train_cat = to_categorical(y_train_encoded)
        y_test_cat = to_categorical(y_test_encoded)
        
        # Normalize features
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        
        return X_train, X_test, y_train_cat, y_test_cat
    
    def train(self, X_train, y_train, X_test, y_test, epochs=None):
        """
        Train the model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            epochs (int): Number of epochs (optional)
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        if self.model is None:
            raise ValueError("Model not provided. Initialize with a model first.")
        
        epochs = epochs or DATA_CONFIG['epochs']
        
        # Setup callbacks
        callbacks = self._setup_callbacks()
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=DATA_CONFIG['batch_size'],
            epochs=epochs,
            verbose=2,
            callbacks=callbacks
        )
        
        return self.history
    
    def _setup_callbacks(self):
        """Setup training callbacks."""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint
        checkpoint_path = os.path.join(PATHS['models_dir'], 'best_model.h5')
        model_checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False
        )
        callbacks.append(model_checkpoint)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard logging
        tensorboard = TensorBoard(
            log_dir=PATHS['logs_dir'],
            histogram_freq=1
        )
        callbacks.append(tensorboard)
        
        return callbacks
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not provided.")
        
        # Evaluate model
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Make predictions
        predictions = self.model.predict(X_test, batch_size=DATA_CONFIG['batch_size'])
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'predictions': predictions
        }
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not provided.")
        
        return self.model.predict(X, batch_size=DATA_CONFIG['batch_size'])
    
    def get_label_mapping(self):
        """
        Get the label encoder mapping.
        
        Returns:
            dict: Label mapping
        """
        return dict(zip(self.label_encoder.classes_, 
                       self.label_encoder.transform(self.label_encoder.classes_)))
    
    def save_predictions(self, predictions, actual, output_path):
        """
        Save predictions to CSV file.
        
        Args:
            predictions (np.ndarray): Model predictions
            actual (np.ndarray): Actual labels
            output_path (str): Output file path
        """
        # Convert predictions to labels
        pred_labels = self.label_encoder.inverse_transform(
            np.argmax(predictions, axis=1)
        )
        actual_labels = self.label_encoder.inverse_transform(
            np.argmax(actual, axis=1)
        )
        
        # Create DataFrame
        results_df = pd.DataFrame({
            'actual': actual_labels,
            'predicted': pred_labels
        })
        
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        return results_df
