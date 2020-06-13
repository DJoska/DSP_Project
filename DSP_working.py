# DSP working file

import librosa
import numpy as np
import matplotlib.pyplot as plt


filename = "C:\\Users\\user-pc\\Downloads\\nsynth-valid.jsonwav.tar\\nsynth-valid.jsonwav\\nsynth-valid\\audio\\bass_electronic_018-023-100.wav"
filename_fl = "C:\\Users\\user-pc\\Downloads\\nsynth-valid.jsonwav.tar\\nsynth-valid.jsonwav\\nsynth-valid\\audio\\keyboard_acoustic_004-053-075.wav"

y, sr = librosa.load(filename)
y_fl, sr_fl = librosa.load(filename_fl)

hop_length = 512

# Separate harmonics and percussives into two waveforms
y_harmonic, y_percussive = librosa.effects.hpss(y)
y_harmonic_fl, y_percussive_fl = librosa.effects.hpss(y_fl)

mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

#temporal averaging
mfcc=np.mean(mfcc,axis=1)

spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)  

#temporally average spectrogram
spectrogram = np.mean(spectrogram, axis = 1)
    
#compute chroma energy
chroma = librosa.feature.chroma_cens(y=y, sr=sr)
#temporally average chroma
chroma = np.mean(chroma, axis = 1)
    
#compute spectral contrast
contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
contrast = np.mean(contrast, axis= 1)

plt.subplot(2,1,1)
plt.plot(y_harmonic, label="Harmonics")
plt.plot(y_percussive, label = "Percussive")
plt.title("Electronic Bass")
plt.legend()
plt.subplot(2,1,2)
plt.plot(y_harmonic_fl, label="Harmonics")
plt.plot(y_percussive_fl, label = "Percussive")
plt.title("Acoustic Flute")
plt.legend()
plt.show()