"""
Prediction script for audio sentiment analysis.
Loads a trained model and predicts emotions from audio files.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

# Add src to path
sys.path.append('src')

from data.data_loader import prepare_single_audio
from models.cnn_model import AudioSentimentCNN
from visualization.plotter import AudioVisualizer
from config.config import DATA_CONFIG


def load_trained_model(model_path, json_path, encoder_path):
    """
    Load a trained model and label encoder.
    
    Args:
        model_path (str): Path to the saved model
        json_path (str): Path to the model JSON file
        encoder_path (str): Path to the label encoder
        
    Returns:
        tuple: (model, label_encoder)
    """
    # Load model
    model = AudioSentimentCNN()
    model.load_model_from_json(json_path, model_path)
    
    # Load label encoder
    label_encoder = joblib.load(encoder_path)
    
    return model, label_encoder


def predict_emotion(audio_path, model, label_encoder, n_mfcc=30):
    """
    Predict emotion from a single audio file.
    
    Args:
        audio_path (str): Path to audio file
        model: Trained model
        label_encoder: Fitted label encoder
        n_mfcc (int): Number of MFCC coefficients
        
    Returns:
        dict: Prediction results
    """
    # Extract features
    features = prepare_single_audio(audio_path, n_mfcc, use_mfcc=True)
    
    # Normalize features (you might need to load the normalization parameters)
    # For now, we'll use basic normalization
    features = (features - np.mean(features)) / (np.std(features) + 1e-8)
    
    # Make prediction
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    
    # Get class name
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    
    # Get all class probabilities
    class_probabilities = {}
    for i, class_name in enumerate(label_encoder.classes_):
        class_probabilities[class_name] = prediction[0][i]
    
    return {
        'predicted_emotion': predicted_label,
        'confidence': confidence,
        'all_probabilities': class_probabilities
    }


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict Audio Sentiment')
    parser.add_argument('--audio_path', type=str, required=True,
                       help='Path to audio file for prediction')
    parser.add_argument('--model_dir', type=str, default='saved_models',
                       help='Directory containing saved model files')
    parser.add_argument('--n_mfcc', type=int, default=30,
                       help='Number of MFCC coefficients')
    parser.add_argument('--visualize', action='store_true',
                       help='Show audio visualization')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AUDIO SENTIMENT ANALYSIS - PREDICTION")
    print("=" * 60)
    
    # Check if audio file exists
    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file '{args.audio_path}' not found!")
        return
    
    # Define model paths
    model_path = os.path.join(args.model_dir, 'audio_sentiment_model.h5')
    json_path = os.path.join(args.model_dir, 'audio_sentiment_model.json')
    encoder_path = os.path.join(args.model_dir, 'label_encoder.pkl')
    
    # Check if model files exist
    for path in [model_path, json_path, encoder_path]:
        if not os.path.exists(path):
            print(f"Error: Model file '{path}' not found!")
            print("Please train the model first using train.py")
            return
    
    # Load model
    print("Loading trained model...")
    model, label_encoder = load_trained_model(model_path, json_path, encoder_path)
    
    # Visualize audio if requested
    if args.visualize:
        visualizer = AudioVisualizer()
        print(f"\nVisualizing audio: {args.audio_path}")
        data, sample_rate = visualizer.plot_audio_waveform(
            args.audio_path, 
            title=f"Audio Waveform - {os.path.basename(args.audio_path)}"
        )
        mfcc = visualizer.plot_mfcc(
            args.audio_path, 
            n_mfcc=args.n_mfcc,
            title=f"MFCC Features - {os.path.basename(args.audio_path)}"
        )
    
    # Make prediction
    print(f"\nPredicting emotion for: {args.audio_path}")
    results = predict_emotion(args.audio_path, model, label_encoder, args.n_mfcc)
    
    # Display results
    print("\n" + "=" * 40)
    print("PREDICTION RESULTS")
    print("=" * 40)
    print(f"Predicted Emotion: {results['predicted_emotion']}")
    print(f"Confidence: {results['confidence']:.4f}")
    
    print(f"\nAll Class Probabilities:")
    print("-" * 30)
    for emotion, prob in sorted(results['all_probabilities'].items(), 
                              key=lambda x: x[1], reverse=True):
        print(f"{emotion:20s}: {prob:.4f}")
    
    print("\n" + "=" * 60)
    print("PREDICTION COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
