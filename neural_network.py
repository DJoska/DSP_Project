# Artificial Neural Network Module

import keras
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras import regularizers, optimizers
from keras import backend as K
from keras.optimizers import SGD
import pandas as pa
from sklearn.model_selection import train_test_split #remove later
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import feature_extraction as fe
import numpy as np
from keras.utils import to_categorical
import time

tic = time.perf_counter()

print(keras.__version__)
y_train = fe.read_instruments("samples_low")
x_train = [fe.get_spectrogram(filename) for filename in fe.get_wav_files("samples_low")]
y_test = fe.read_instruments("samples_tiny")
x_test = [fe.get_spectrogram(filename) for filename in fe.get_wav_files("samples_tiny")]

#url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
#dataset = pa.read_csv(url, names=names)
#print(dataset.head())
#X = dataset.iloc[:, :-1].values
#y = dataset.iloc[:, 4].values
#for i in range(len(y)):
#    if y[i] == "Iris-setosa":
#        y[i] = 0
#    if y[i] == "Iris-versicolor":
#        y[i] = 1
 #   if y[i] == "Iris-virginica":
 #       y[i] = 2

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.40)

print("X length:")
print(len(x_train))
print("and Y length:")
print(len(y_train))
x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)
y_train = to_categorical(y_train)
y_test= to_categorical(y_test)
x_train = x_train.reshape(x_train.shape[0], 128, 173, 1)
x_test = x_test.reshape(x_test.shape[0], 128, 173, 1)
print(x_train[0].shape)

toc = time.perf_counter()
time_data = round((toc-tic),4)

tic = time.perf_counter()

input_shape = x_train[0].shape
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128,173,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(11, activation='softmax'))

opt = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)

model.compile(optimizer=opt,loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()

# fit the keras model on the dataset
model.fit(x_train, y_train, epochs=20, batch_size=100)

toc = time.perf_counter()
time_train = round((toc-tic),4)
tic = time.perf_counter()

# evaluate the keras model
_, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

toc = time.perf_counter()
time_test = round((toc-tic),4)

print("Data collection time: ", time_data, "s")
print("Training time: ", time_train, "s")
print("Testing time: ", time_test, "s")