"""
Data loading and preprocessing module for audio sentiment analysis.
Handles RAVDESS dataset loading and feature extraction.
"""

import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from config.config import DATA_CONFIG, EMOTION_MAPPING, GENDER_MAPPING


class RAVDESSDataLoader:
    """Data loader for RAVDESS emotional speech audio dataset."""
    
    def __init__(self, data_path):
        """
        Initialize the data loader.
        
        Args:
            data_path (str): Path to the RAVDESS dataset directory
        """
        self.data_path = data_path
        self.df = None
        
    def load_metadata(self):
        """
        Load and create metadata for the RAVDESS dataset.
        
        Returns:
            pd.DataFrame: DataFrame containing file paths, emotions, and labels
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path {self.data_path} does not exist")
            
        dir_list = os.listdir(self.data_path)
        dir_list.sort()
        
        emotion = []
        gender = []
        path = []
        
        for actor_dir in dir_list:
            actor_path = os.path.join(self.data_path, actor_dir)
            if not os.path.isdir(actor_path):
                continue
                
            files = os.listdir(actor_path)
            for file in files:
                if not file.endswith('.wav'):
                    continue
                    
                # Parse filename: 03-01-01-01-01-01-01.wav
                parts = file.split('.')[0].split('-')
                if len(parts) >= 7:
                    emotion_code = int(parts[2])
                    actor_id = int(parts[6])
                    
                    # Map emotion code to emotion name
                    emotion_name = EMOTION_MAPPING.get(emotion_code, 'unknown')
                    emotion.append(emotion_name)
                    
                    # Determine gender based on actor ID
                    gender_name = GENDER_MAPPING['even'] if actor_id % 2 == 0 else GENDER_MAPPING['odd']
                    gender.append(gender_name)
                    
                    # Full file path
                    file_path = os.path.join(actor_path, file)
                    path.append(file_path)
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'emotion': emotion,
            'gender': gender,
            'path': path
        })
        
        # Create combined labels
        self.df['labels'] = self.df['gender'] + '_' + self.df['emotion']
        self.df['source'] = 'RAVDESS'
        
        # Drop individual columns to keep only labels and path
        self.df = self.df[['labels', 'path', 'source']]
        
        return self.df
    
    def get_data_info(self):
        """
        Get information about the loaded dataset.
        
        Returns:
            dict: Dataset information including shape and label counts
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_metadata() first.")
            
        return {
            'shape': self.df.shape,
            'label_counts': self.df['labels'].value_counts().to_dict(),
            'unique_labels': self.df['labels'].unique().tolist()
        }
    
    def save_metadata(self, output_path):
        """
        Save metadata to CSV file.
        
        Args:
            output_path (str): Path to save the CSV file
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_metadata() first.")
            
        self.df.to_csv(output_path, index=False)
        print(f"Metadata saved to {output_path}")


def prepare_audio_data(df, n_mfcc, aug=False, use_mfcc=True):
    """
    Extract audio features (MFCC or Mel-spectrogram) from audio files.
    
    Args:
        df (pd.DataFrame): DataFrame with audio file paths
        n_mfcc (int): Number of MFCC coefficients
        aug (bool): Whether to apply data augmentation
        use_mfcc (bool): Whether to use MFCC (True) or Mel-spectrogram (False)
        
    Returns:
        np.ndarray: Extracted features array
    """
    sampling_rate = DATA_CONFIG['sampling_rate']
    audio_duration = DATA_CONFIG['audio_duration']
    n_melspec = DATA_CONFIG['n_melspec']
    
    input_length = sampling_rate * audio_duration
    X = np.empty(shape=(df.shape[0], n_mfcc, 216, 1))
    
    for cnt, fname in enumerate(tqdm(df.path, desc="Extracting features")):
        try:
            # Load audio file
            data, _ = librosa.load(
                fname, 
                sr=sampling_rate,
                duration=audio_duration,
                offset=0.5
            )
            
            # Random offset / Padding
            if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.randint(max_offset)
                data = data[offset:(input_length + offset)]
            else:
                if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.randint(max_offset)
                else:
                    offset = 0
                data = np.pad(
                    data, 
                    (offset, int(input_length) - len(data) - offset), 
                    "constant"
                )
            
            # Data augmentation (placeholder - implement speedNpitch function)
            if aug:
                # data = speedNpitch(data)  # TODO: Implement augmentation
                pass
            
            # Feature extraction
            if use_mfcc:
                # MFCC extraction
                mfcc = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=n_mfcc)
                mfcc = np.expand_dims(mfcc, axis=-1)
                X[cnt] = mfcc
            else:
                # Mel-spectrogram
                melspec = librosa.feature.melspectrogram(data, n_mels=n_melspec)
                logspec = librosa.amplitude_to_db(melspec)
                logspec = np.expand_dims(logspec, axis=-1)
                X[cnt] = logspec
                
        except Exception as e:
            print(f"Error processing file {fname}: {e}")
            # Fill with zeros if error occurs
            X[cnt] = np.zeros((n_mfcc, 216, 1))
    
    return X


def prepare_single_audio(file_path, n_mfcc, use_mfcc=True):
    """
    Extract features from a single audio file.
    
    Args:
        file_path (str): Path to the audio file
        n_mfcc (int): Number of MFCC coefficients
        use_mfcc (bool): Whether to use MFCC (True) or Mel-spectrogram (False)
        
    Returns:
        np.ndarray: Extracted features array
    """
    sampling_rate = DATA_CONFIG['sampling_rate']
    audio_duration = DATA_CONFIG['audio_duration']
    n_melspec = DATA_CONFIG['n_melspec']
    
    input_length = sampling_rate * audio_duration
    X = np.empty(shape=(1, n_mfcc, 216, 1))
    
    try:
        # Load audio file
        data, _ = librosa.load(
            file_path,
            sr=sampling_rate,
            duration=audio_duration,
            offset=0.5
        )
        
        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length + offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(
                data,
                (offset, int(input_length) - len(data) - offset),
                "constant"
            )
        
        # Feature extraction
        if use_mfcc:
            # MFCC extraction
            mfcc = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=n_mfcc)
            mfcc = np.expand_dims(mfcc, axis=-1)
            X[0] = mfcc
        else:
            # Mel-spectrogram
            melspec = librosa.feature.melspectrogram(data, n_mels=n_melspec)
            logspec = librosa.amplitude_to_db(melspec)
            logspec = np.expand_dims(logspec, axis=-1)
            X[0] = logspec
            
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        X[0] = np.zeros((n_mfcc, 216, 1))
    
    return X
