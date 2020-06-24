# DSP - Feature Extraction Module

import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time

def read_wav_names(path):
    """
    Returns an array of the names of all .wav files in the /samples/ directory, minus the .wav
    """
    print("Reading files in ", path, "...")
    folder = os.path.join(os.path.dirname(__file__),path)
    paths = os.listdir(folder)
    wav_names = [f[:-4] for f in paths if f.endswith(".wav")]
    return(wav_names)

def read_instruments(path):
    """
    Reads the json file in the /samples/ directory and returns ints representing instrument families
    """
    folder = os.path.join(os.path.dirname(__file__),path)
    json_path = os.path.join(folder,"examples.json")
    wav_names = read_wav_names(path)
    # Extract dataframe
    df = pd.read_json(json_path)
    instruments = [df[wav_name]["instrument_family"] for wav_name in wav_names]
    return(instruments)

def get_wav_files(path):
    """
    Returns the directories of all .wav audio files within the /samples/ directory
    """
    print("Retreiving wav files from ", path, "...")
    folder = os.path.join(os.path.dirname(__file__),path)
    wav_names = read_wav_names(path)
    wav_files = [os.path.join(folder,(f+".wav")) for f in wav_names]
    print("Successfully found wav files.")
    #print(wav_files)
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

def get_spectrogram(filename):
    """
    Gets the mel spectrogram of a given audio file
    """
    y, sr = librosa.load(filename)
    spectrogram = np.array(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000))
    print("Extracted spectrogram from "+filename)
    return(spectrogram)

def get_1d_spectrogram(filename):
    """
    Gets the mel spectrogram of a given audio file, returns 1d array
    """
    y, sr = librosa.load(filename)
    spectrogram = np.array(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000))
    print("Extracted spectrogram from "+filename)
    return(spectrogram.flatten())

if __name__ == "__main__":
    spec = get_1d_spectrogram(get_wav_files("testing")[0])