# AI Mood Analyer

An audio based emotion and sentiment classification from audio speech using MFCC features and Convolutional Neural Networks (CNN). This system uses advanced signal processing techniques (MFCC features) combined with a CNN architecture to automatically detect and classify emotions in human speech.

## What This Project Does

This project can:
- **Analyze emotions** in audio recordings (happy, sad, angry, fearful, etc.)
- **Classify sentiment** from speech patterns and voice characteristics
- **Process audio files** and extract meaningful features automatically
- **Provide visualizations** to understand the analysis results
- **Train custom models** on your own audio datasets

## Practical Applications

- **Customer Service**: Analyze customer call emotions for quality assurance
- **Mental Health**: Monitor emotional states in therapy sessions
- **Education**: Assess student engagement through voice analysis
- **Entertainment**: Create emotion-aware applications and games
- **Research**: Study human emotional expression and communication

## Project Structure

```
Audio-Sentiment-Analysis/
├── src/
│   ├── data/
│   │   └── data_loader.py          # Data loading and preprocessing
│   ├── models/
│   │   ├── cnn_model.py            # CNN model architecture
│   │   └── trainer.py              # Model training and evaluation
│   ├── visualization/
│   │   └── plotter.py              # Visualization utilities
│   └── utils/
├── config/
│   └── config.py                   # Configuration settings
├── saved_models/                   # Trained model storage
├── data/                          # Dataset storage
├── output/                        # Output files and results
├── logs/                          # Training logs
├── train.py                       # Main training script
├── predict.py                     # Prediction script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Features

- **Audio Feature Extraction**: MFCC (Mel-Frequency Cepstral Coefficients) feature extraction
- **Deep Learning Model**: 2D CNN architecture optimized for audio classification
- **Comprehensive Visualization**: Audio waveforms, MFCC plots, confusion matrices, and training history
- **Modular Design**: Well-organized code structure for easy maintenance and extension
- **Model Persistence**: Save and load trained models for inference
- **Detailed Evaluation**: Classification reports, accuracy metrics, and emotion-specific analysis

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Audio-Sentiment-Analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the RAVDESS dataset:
   - Visit the [RAVDESS dataset page](https://zenodo.org/record/1188976)
   - Download the audio files
   - Extract to a directory (e.g., `data/ravdess/`)

## Usage

### Training

Train the model on the RAVDESS dataset:

```bash
python train.py --data_path /path/to/dataset
```

**Training Arguments:**
- `--data_path`: Path to RAVDESS dataset directory (required)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 16)
- `--n_mfcc`: Number of MFCC coefficients (default: 30)
- `--augment`: Use data augmentation (flag)

### Prediction

Predict emotion from an audio file:

```bash
python predict.py --audio_path /path/to/audio.wav 
```

**Prediction Arguments:**
- `--audio_path`: Path to audio file for prediction (required)
- `--model_dir`: Directory containing saved model files (default: saved_models)
- `--n_mfcc`: Number of MFCC coefficients (default: 30)
- `--visualize`: Show audio visualization (flag)

## Dataset

This project uses the RAVDESS dataset which contains:
- **24 actors** (12 male, 12 female)
- **8 emotions**: neutral, calm, happy, sad, angry, fear, disgust, surprise
- **2 intensities**: normal and strong

## Configuration

All hyperparameters and settings are centralized in `config/config.py`:

- **Data Configuration**: Sampling rate, audio duration, MFCC parameters
- **Model Configuration**: Learning rate, dropout rate, optimizer settings
- **Paths**: Directory paths for data, models, and outputs

## Results

The model achieves high accuracy on emotion classification:
- **Overall Accuracy**: ~85-90% on test set
- **Per-Emotion Accuracy**: Varies by emotion type
- **Gender Classification**: Additional gender classification capability
