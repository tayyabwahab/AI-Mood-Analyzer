"""
Example usage script for Audio Sentiment Analysis.
Demonstrates how to use the modular components.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add src to path
sys.path.append('src')

from data.data_loader import RAVDESSDataLoader, prepare_audio_data
from models.cnn_model import AudioSentimentCNN
from models.trainer import ModelTrainer
from visualization.plotter import AudioVisualizer
from config.config import DATA_CONFIG


def example_data_loading():
    """Example of loading and exploring data."""
    print("=== Data Loading Example ===")
    
    # Note: You need to have the RAVDESS dataset downloaded
    data_path = "data/ravdess/audio_speech_actors_01-24/"
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        print("Please download the RAVDESS dataset and update the path")
        return None
    
    # Load data
    loader = RAVDESSDataLoader(data_path)
    df = loader.load_metadata()
    
    # Display info
    info = loader.get_data_info()
    print(f"Dataset shape: {info['shape']}")
    print(f"Unique labels: {info['unique_labels']}")
    
    return df


def example_model_creation():
    """Example of creating and building a model."""
    print("\n=== Model Creation Example ===")
    
    # Create model
    model = AudioSentimentCNN(n_mfcc=30, nclass=16)
    model.build_model()
    model.compile_model()
    
    # Display model summary
    print("Model Architecture:")
    model.get_model_summary()
    
    return model


def example_visualization():
    """Example of visualization capabilities."""
    print("\n=== Visualization Example ===")
    
    visualizer = AudioVisualizer()
    
    # Example audio path (replace with actual path)
    audio_path = "data/example_audio.wav"
    
    if os.path.exists(audio_path):
        # Plot audio waveform
        data, sample_rate = visualizer.plot_audio_waveform(
            audio_path, 
            title="Example Audio Waveform"
        )
        
        # Plot MFCC features
        mfcc = visualizer.plot_mfcc(
            audio_path, 
            n_mfcc=30,
            title="Example MFCC Features"
        )
    else:
        print(f"Example audio file not found at {audio_path}")
        print("Skipping visualization example")


def example_training_workflow():
    """Example of complete training workflow."""
    print("\n=== Training Workflow Example ===")
    
    # This is a simplified example - in practice, you'd use the train.py script
    print("For complete training, use: python train.py --data_path /path/to/dataset")
    
    # Example of creating a small dataset for demonstration
    print("Creating example data...")
    
    # Create dummy data for demonstration
    n_samples = 100
    n_mfcc = 30
    X_dummy = np.random.randn(n_samples, n_mfcc, 216, 1)
    y_dummy = np.random.choice(['female_happy', 'male_sad', 'female_angry', 'male_calm'], n_samples)
    
    # Create model
    model = AudioSentimentCNN(n_mfcc=n_mfcc)
    model.build_model()
    model.compile_model()
    
    # Create trainer
    trainer = ModelTrainer(model.model)
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(X_dummy, y_dummy)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print("Model ready for training!")


def main():
    """Main example function."""
    print("Audio Sentiment Analysis - Example Usage")
    print("=" * 50)
    
    # Example 1: Data loading
    df = example_data_loading()
    
    # Example 2: Model creation
    model = example_model_creation()
    
    # Example 3: Visualization
    example_visualization()
    
    # Example 4: Training workflow
    example_training_workflow()
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("For full functionality, ensure you have:")
    print("1. RAVDESS dataset downloaded")
    print("2. All dependencies installed (pip install -r requirements.txt)")
    print("3. Run train.py for training")
    print("4. Run predict.py for inference")


if __name__ == "__main__":
    main()
