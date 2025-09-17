"""
Visualization module for audio sentiment analysis.
Handles plotting of results, confusion matrices, and audio analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import librosa
import librosa.display
from sklearn.metrics import confusion_matrix, classification_report
import IPython.display as ipd


class AudioVisualizer:
    """Handles visualization of audio data and model results."""
    
    def __init__(self, figsize=(15, 10)):
        """
        Initialize the visualizer.
        
        Args:
            figsize (tuple): Default figure size
        """
        self.figsize = figsize
        plt.style.use('default')
        
    def plot_audio_waveform(self, audio_path, title="Audio Waveform"):
        """
        Plot audio waveform.
        
        Args:
            audio_path (str): Path to audio file
            title (str): Plot title
        """
        data, sample_rate = librosa.load(audio_path)
        
        plt.figure(figsize=self.figsize)
        librosa.display.waveshow(data, sr=sample_rate)
        plt.title(title, size=20)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.show()
        
        return data, sample_rate
    
    def plot_mfcc(self, audio_path, n_mfcc=13, title="MFCC Features"):
        """
        Plot MFCC features of an audio file.
        
        Args:
            audio_path (str): Path to audio file
            n_mfcc (int): Number of MFCC coefficients
            title (str): Plot title
        """
        data, sample_rate = librosa.load(audio_path, duration=2.5, sr=22050*2, offset=0.5)
        mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=n_mfcc)
        
        plt.figure(figsize=self.figsize)
        librosa.display.specshow(mfcc, x_axis='time')
        plt.ylabel('MFCC')
        plt.colorbar()
        plt.title(title, size=20)
        plt.tight_layout()
        plt.show()
        
        return mfcc
    
    def plot_label_distribution(self, df, column='labels', title="Label Distribution"):
        """
        Plot distribution of labels.
        
        Args:
            df (pd.DataFrame): DataFrame with labels
            column (str): Column name containing labels
            title (str): Plot title
        """
        plt.figure(figsize=(20, 6))
        df[column].value_counts().plot(kind='bar')
        plt.title(title, size=20)
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print(f"Label counts:\n{df[column].value_counts()}")
    
    def plot_training_history(self, history, title="Training History"):
        """
        Plot training history (loss and accuracy).
        
        Args:
            history: Keras training history object
            title (str): Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        
        plt.suptitle(title, size=16)
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, 
                            figsize=(10, 7), fontsize=14):
        """
        Plot confusion matrix.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            class_names (list): List of class names
            figsize (tuple): Figure size
            fontsize (int): Font size
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        
        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap='Blues')
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        
        heatmap.yaxis.set_ticklabels(
            heatmap.yaxis.get_ticklabels(), 
            rotation=0, 
            ha='right', 
            fontsize=fontsize
        )
        heatmap.xaxis.set_ticklabels(
            heatmap.xaxis.get_ticklabels(), 
            rotation=45, 
            ha='right', 
            fontsize=fontsize
        )
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def plot_gender_comparison(self, audio_paths, labels, title="Gender Comparison"):
        """
        Plot MFCC comparison between male and female speakers.
        
        Args:
            audio_paths (list): List of audio file paths
            labels (list): List of corresponding labels
            title (str): Plot title
        """
        plt.figure(figsize=(20, 15))
        
        for i, (path, label) in enumerate(zip(audio_paths, labels)):
            data, sample_rate = librosa.load(path, duration=2.5, sr=22050*2, offset=0.5)
            mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=0)
            
            plt.subplot(2, 2, i+1)
            plt.plot(mfcc_mean, label=label)
            plt.title(f'MFCC Features - {label}')
            plt.xlabel('Time Frames')
            plt.ylabel('MFCC Coefficients')
            plt.legend()
        
        plt.suptitle(title, size=16)
        plt.tight_layout()
        plt.show()
    
    def print_classification_report(self, y_true, y_pred, class_names):
        """
        Print detailed classification report.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            class_names (list): List of class names
        """
        report = classification_report(y_true, y_pred, target_names=class_names)
        print("Classification Report:")
        print("=" * 50)
        print(report)
    
    def plot_emotion_accuracy(self, results_df, title="Emotion Classification Accuracy"):
        """
        Plot accuracy for each emotion class.
        
        Args:
            results_df (pd.DataFrame): DataFrame with actual and predicted columns
            title (str): Plot title
        """
        # Calculate accuracy for each emotion
        emotion_accuracy = []
        emotions = results_df['actual'].unique()
        
        for emotion in emotions:
            emotion_data = results_df[results_df['actual'] == emotion]
            accuracy = (emotion_data['actual'] == emotion_data['predicted']).mean()
            emotion_accuracy.append(accuracy)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(emotions, emotion_accuracy)
        plt.title(title, size=16)
        plt.xlabel('Emotion')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, emotion_accuracy):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return dict(zip(emotions, emotion_accuracy))
    
    def play_audio(self, audio_path):
        """
        Play audio file in Jupyter notebook.
        
        Args:
            audio_path (str): Path to audio file
        """
        return ipd.Audio(audio_path)
