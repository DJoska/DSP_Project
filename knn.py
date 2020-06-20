# DSP - Training Module

import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pa
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split #remove later
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import feature_extraction as fe

# Hyperparameter variables go here
k = -1
k_lim = 20
leaf_size = -1
l_lim = 30
p = -1

tic = time.perf_counter()

# Real data for X and y
path = "samples_huge"
pathvali = "validatedata"
pathtest = "testdata"
y_train = fe.read_instruments(path)
X_train = [fe.get_features(filename) for filename in fe.get_wav_files(path)]
print("Training data loaded in.")
#print("X length:")
#print(len(X))
#print("and Y length:")
#print(len(y))

# Data collection
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
#X_train, X_vali, y_train, y_vali = train_test_split(X_train, y_train, test_size=0.25)
y_vali = fe.read_instruments(pathvali)
X_vali = [fe.get_features(filename) for filename in fe.get_wav_files(pathvali)]
print("Validate data loaded in.")
#X_vali, X_test, y_vali, y_test = train_test_split(X_vali, y_vali, test_size=0.30)
y_test =  fe.read_instruments(pathtest)
X_test = [fe.get_features(filename) for filename in fe.get_wav_files(pathtest)]
print("Test data loaded in.")
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_vali = scaler.transform(X_vali)
X_test = scaler.transform(X_test)
print("X_train: ", len(X_train), "\nX_vali: ", len(X_vali), "\nX_test: ", len(X_test))
toc = time.perf_counter()
time_data = round((toc-tic),4)

tic = time.perf_counter()
#Validation cycle (k)
accuracy_k = []
accMax = 0
for i in range (1, k_lim):
    kmeans = KNeighborsClassifier(n_neighbors=i)
    kmeans.fit(X_train, y_train)
    y_pred = kmeans.predict(X_vali)
    accTemp = (np.sum(y_pred == y_vali)/len(y_vali))*100
    if(accTemp > accMax):
        k = i
        accMax = accTemp
    accuracy_k.append(round(accTemp,2))

#Validation cycle (leaves)
accuracy_l = []
accMax = 0
for i in range (1, l_lim):
    kmeans = KNeighborsClassifier(n_neighbors=k, leaf_size=i) #Should I use the optimal k value here as well?
    kmeans.fit(X_train, y_train)
    y_pred = kmeans.predict(X_vali)
    accTemp = (np.sum(y_pred == y_vali)/len(y_vali))*100
    if(accTemp > accMax):
        leaf_size = i
        accMax = accTemp
    accuracy_l.append(round(accTemp,2))

#Validation cycle (p)
accuracy_p = []
kmeans = KNeighborsClassifier(n_neighbors=k, leaf_size=leaf_size, p=1)
kmeans.fit(X_train, y_train)
y_pred = kmeans.predict(X_vali)
acc1 = (np.sum(y_pred == y_vali)/len(y_vali))*100
accuracy_p.append(round(acc1, 2))
kmeans = KNeighborsClassifier(n_neighbors=k, leaf_size=leaf_size, p=2)
kmeans.fit(X_train, y_train)
y_pred = kmeans.predict(X_vali)
acc2 = (np.sum(y_pred == y_vali)/len(y_vali))*100
accuracy_p.append(round(acc2, 2))
if (acc1 > acc2):
    p = 1
else:
    p = 2
if (acc1 == acc2):
    print("Distance method was inconsequential")

#Analysis based on optimised hyperparameters
print("Fully trained. Selected hyperparameters:\nk: ", k, "\nLeaf size: ", leaf_size, "\nMeasurement distance: ", end='')
if (p == 1):
    print("Manhatten")
elif (p == 2):
    print("Euclidean")
else:
    print ("Error undf")
knn = KNeighborsClassifier(n_neighbors=k, leaf_size=leaf_size, p=p)
knn.fit(X_train, y_train)
toc = time.perf_counter()
time_train = round((toc-tic),4)
tic = time.perf_counter()
y_pred = knn.predict(X_test)
toc = time.perf_counter()
time_test = round((toc-tic),4)

# Results display
print("\nFully tested. Displaying results:")
# Print accuracy results for various k values
plt.figure(figsize=(6, 6))
plt.plot(range(1, k_lim), accuracy_k, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Accuracy per K value')
plt.xlabel('K Value')
plt.ylabel('Tested Accuracy')
plt.show()

# Measurement Reports
print(classification_report(y_test, y_pred))
print("Data collection time: ", time_data, "s")
print("Training time: ", time_train, "s")
print("Testing time: ", time_test, "s")

pie_labels = 'Feature extraction', 'Training'#, 'Testing'
time_sum = time_data+time_train#+time_test
pie_vals = [time_data/time_sum, time_train/time_sum]#, time_test/time_sum]
fig1, ax1 = plt.subplots()
ax1.pie(pie_vals, labels=pie_labels, startangle=90)
ax1.axis('equal')
plt.show()