"""
Main training script for audio sentiment analysis.
This script trains a CNN model on RAVDESS dataset for emotion classification.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

from data.data_loader import RAVDESSDataLoader, prepare_audio_data
from models.cnn_model import AudioSentimentCNN
from models.trainer import ModelTrainer
from visualization.plotter import AudioVisualizer
from config.config import DATA_CONFIG, PATHS


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Audio Sentiment Analysis Model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to RAVDESS dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--n_mfcc', type=int, default=30,
                       help='Number of MFCC coefficients')
    parser.add_argument('--augment', action='store_true',
                       help='Use data augmentation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AUDIO SENTIMENT ANALYSIS - MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    print("\n1. Loading and preparing data...")
    data_loader = RAVDESSDataLoader(args.data_path)
    df = data_loader.load_metadata()
    
    # Save metadata
    data_loader.save_metadata(os.path.join(PATHS['data_dir'], 'metadata.csv'))
    
    # Display data info
    info = data_loader.get_data_info()
    print(f"Dataset shape: {info['shape']}")
    print(f"Number of unique labels: {len(info['unique_labels'])}")
    
    # Visualize label distribution
    visualizer = AudioVisualizer()
    visualizer.plot_label_distribution(df, title="RAVDESS Dataset Label Distribution")
    
    # Step 2: Extract features
    print("\n2. Extracting MFCC features...")
    X = prepare_audio_data(
        df, 
        n_mfcc=args.n_mfcc, 
        aug=args.augment, 
        use_mfcc=True
    )
    print(f"Feature shape: {X.shape}")
    
    # Step 3: Build and train model
    print("\n3. Building and training model...")
    
    # Create model
    model = AudioSentimentCNN(n_mfcc=args.n_mfcc)
    model.build_model()
    model.compile_model()
    
    print("Model architecture:")
    model.get_model_summary()
    
    # Create trainer
    trainer = ModelTrainer(model.model)
    
    # Prepare data for training
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, df['labels'])
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Train model
    history = trainer.train(X_train, y_train, X_test, y_test, epochs=args.epochs)
    
    # Step 4: Evaluate model
    print("\n4. Evaluating model...")
    results = trainer.evaluate(X_test, y_test)
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test Loss: {results['loss']:.4f}")
    
    # Plot training history
    visualizer.plot_training_history(history, title="Model Training History")
    
    # Generate predictions and confusion matrix
    predictions = results['predictions']
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Get class names
    class_names = trainer.label_encoder.classes_
    
    # Plot confusion matrix
    visualizer.plot_confusion_matrix(
        y_true, y_pred, class_names,
        title="Confusion Matrix - All Classes"
    )
    
    # Print classification report
    visualizer.print_classification_report(y_true, y_pred, class_names)
    
    # Save predictions
    results_df = trainer.save_predictions(
        predictions, y_test, 
        os.path.join(PATHS['output_dir'], 'predictions.csv')
    )
    
    # Plot emotion-specific accuracy
    visualizer.plot_emotion_accuracy(results_df, title="Emotion Classification Accuracy")
    
    # Step 5: Save model
    print("\n5. Saving model...")
    model_path = os.path.join(PATHS['models_dir'], 'audio_sentiment_model.h5')
    model.save_model(model_path)
    
    json_path = os.path.join(PATHS['models_dir'], 'audio_sentiment_model.json')
    model.save_model_json(json_path)
    
    # Save label encoder
    import joblib
    encoder_path = os.path.join(PATHS['models_dir'], 'label_encoder.pkl')
    joblib.dump(trainer.label_encoder, encoder_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Model JSON saved to: {json_path}")
    print(f"Label encoder saved to: {encoder_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
