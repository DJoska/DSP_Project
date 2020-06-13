# DSP working file

import librosa
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split #remove later
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

#filename = "E:\\Uni Work\\DSP\\Data\\nsynth-valid\\audio\\bass_electronic_018-023-100.wav"
#filename_fl = "E:\\Uni Work\\DSP\\Data\\nsynth-valid\\audiokeyboard_acoustic_004-053-075.wav"

#y, sr = librosa.load(filename)
#y_fl, sr_fl = librosa.load(filename_fl)

#hop_length = 512

# Separate harmonics and percussives into two waveforms
#y_harmonic, y_percussive = librosa.effects.hpss(y)
#y_harmonic_fl, y_percussive_fl = librosa.effects.hpss(y_fl)

#mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

#temporal averaging
#mfcc=np.mean(mfcc,axis=1)

#spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)  

#temporally average spectrogram
#spectrogram = np.mean(spectrogram, axis = 1)
    
#compute chroma energy
#chroma = librosa.feature.chroma_cens(y=y, sr=sr)
#temporally average chroma
#chroma = np.mean(chroma, axis = 1)
    
#compute spectral contrast
#contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
#contrast = np.mean(contrast, axis= 1)

#plt.subplot(2,1,1)
#plt.plot(y_harmonic, label="Harmonics")
#plt.plot(y_percussive, label = "Percussive")
#plt.title("Electronic Bass")
#plt.legend()
#plt.subplot(2,1,2)
#plt.plot(y_harmonic_fl, label="Harmonics")
#plt.plot(y_percussive_fl, label = "Percussive")
#plt.title("Acoustic Flute")
#plt.legend()
#plt.show()

#Hyperparameter variables go here
k = 5
k_lim = 20

#Dummy data set until we get the feature fully solved
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pa.read_csv(url, names=names)
#print(dataset.head())
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Input data plotting
#fig = plt.figure(figsize=(6,6))
#plt.scatter(X_train, y_train, color='c', label='train')
#plt.scatter(X_test, y_test, color='m', label='test')
#plt.xlabel('x')
#plt.ylabel('y')
#lt.legend()
#plt.show()


#Training phase (basic)
knn_basic = KNeighborsClassifier(n_neighbors=k)
knn_basic.fit(X_train, y_train)
y_pred_basic = knn_basic.predict(X_test)

#Validation cycle (manual)
accuracy = []
accMax = 0
for i in range (1, k_lim):
    kmeans = KNeighborsClassifier(n_neighbors=i)
    kmeans.fit(X_train, y_train)
    y_pred = kmeans.predict(X_test)
    if((np.sum(y_pred == y_test)/len(y_test))*100 > accMax):
        k = i
    accuracy.append(round((np.sum(y_pred == y_test)/len(y_test))*100,2))

#Analysis based on optimised hyperparameters
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

#Validation cycle (FANCY AUTOMATIC)
leaf_size = list(range(1,50))
n_neighbors = list(range(1,k_lim))
p=[1,2]
#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
print("Attempting optimisation...")
knn_setup = KNeighborsClassifier()
grid = GridSearchCV(knn_setup, hyperparameters, cv=10)
knn_optimal = grid.fit(X_train, y_train)
print('Best leaf_size:', knn_optimal.best_estimator_.get_params()['leaf_size'])
if (knn_optimal.best_estimator_.get_params()['p'] == 1):
    print("Best distance measure: Manhatten")
elif (knn_optimal.best_estimator_.get_params()['p'] == 2):
    print("Best distance measrue: Euclidean")
print('Best n_neighbors:', knn_optimal.best_estimator_.get_params()['n_neighbors'])
y_pred_optimal = knn_optimal.predict(X_test)

#Results display
print("Displaying results:")
#Print accuracy results for various k values
plt.figure(figsize=(6, 6))
plt.plot(range(1, k_lim), accuracy, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Accuracy per K value')
plt.xlabel('K Value')
plt.ylabel('Tested Accuracy')
plt.show()
#Accuracy Reports
print(classification_report(y_test, y_pred_basic))
print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred_optimal))
#rint(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))