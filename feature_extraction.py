# DSP - Feature Extraction Module

import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def read_wav_names():
    """
    Returns an array of the names of all .wav files in the /samples/ directory, minus the .wav
    """
    folder = os.path.join(os.path.dirname(__file__),"samples")
    paths = os.listdir(folder)
    wav_names = [f[:-4] for f in paths if f.endswith(".wav")]
    return(wav_names)

def read_instruments():
    """
    Reads the json file in the /samples/ directory and returns ints representing instrument families
    """
    folder = os.path.join(os.path.dirname(__file__),"samples")
    json_path = os.path.join(folder,"examples.json")
    wav_names = read_wav_names()
    # Extract dataframe
    df = pd.read_json(json_path)
    instruments = [df[wav_name]["instrument_family"] for wav_name in wav_names]
    return(instruments)

def get_wav_files():
    """
    Returns the directories of all .wav audio files within the /samples/ directory
    """
    folder = os.path.join(os.path.dirname(__file__),"samples")
    wav_names = read_wav_names()
    wav_files = [os.path.join(folder,(f+".wav")) for f in wav_names]
    print("Found wav files:")
    print(wav_files)
    return wav_files

def get_features(filename):
    """
    Returns the features (chroma, mpcc, etc) from the given audio file
    """
    y, sr = librosa.load(filename)

    #hop_length = 512

    # Separate harmonics and percussives into two waveforms
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Get harmonic and percussive means
    harmonic_mean = np.array(np.mean(y_harmonic))
    percussive_mean = np.array(np.mean(y_percussive))
    means= [harmonic_mean, percussive_mean]

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc,axis=1)

    # Spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
    spectrogram = np.mean(spectrogram, axis = 1)
    
    # Chroma Energy
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma = np.mean(chroma, axis = 1)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast = np.mean(contrast, axis= 1)

    # Concatenate all arrays
    features = np.concatenate((means, mfcc, spectrogram, chroma, contrast))

    print("Extracted features from "+filename)
    return(features)
