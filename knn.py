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
k = -1 #k no. of neighbours. Variables set to -1 have no yet been initialised
k_lim = 20
leaf_size = -1 #Leaf size for model
l_lim = 30
p = -1 #Distance measurement "p"

tic = time.perf_counter() #Start timer for feature extraction

# Real data for X and y
path = "samples_large" #Directory for training data
pathtest = "testdata" #Directory for "test data" (will be split into validate and test)
y_train = fe.read_instruments(path)
X_train = [fe.get_features(filename) for filename in fe.get_wav_files(path)]
print("Training data loaded in.")

# Data collection
y_vali = fe.read_instruments(pathtest)
X_vali = [fe.get_features(filename) for filename in fe.get_wav_files(pathtest)]
print("Validate data loaded in.")

#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20)
#X_train, X_vali, y_train, y_vali = train_test_split(X_train, y_train, test_size=0.25)
# This block of code is used when the training data is too small to warrant use of full test set
# In this case we instead partition the training set 60/20/20

X_vali, X_test, y_vali, y_test = train_test_split(X_vali, y_vali, test_size=0.30) #Split validation/test 70/30
print("Test data loaded in.")
scaler = StandardScaler() #Perform some scaling on the data
# Note: Does not affect accuracy but its good practice so its been left in
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_vali = scaler.transform(X_vali)
X_test = scaler.transform(X_test)
print("X_train: ", len(X_train), "\nX_vali: ", len(X_vali), "\nX_test: ", len(X_test))
toc = time.perf_counter()
time_data = round((toc-tic),4) #Record feature extraction (and data partitioning time)
# Effectively the "time taken to prepare data"

tic = time.perf_counter() #Start training clock
#Validation cycle (k)
accuracy_k = []
accMax = 0
for i in range (1, k_lim):
    kmeans = KNeighborsClassifier(n_neighbors=i) #Set up a model for given k
    kmeans.fit(X_train, y_train)
    y_pred = kmeans.predict(X_vali) #Predict using validation set
    accTemp = (np.sum(y_pred == y_vali)/len(y_vali))*100 #Accuracy is correct guesses/total
    if(accTemp > accMax): #If this is the highest accuracy noted, update the optimal k value (used later)
        k = i
        accMax = accTemp #Update max
    accuracy_k.append(round(accTemp,2)) #Track accuracy history

#Validation cycle (leaves)
accuracy_l = []
accMax = 0
for i in range (1, l_lim):
    kmeans = KNeighborsClassifier(n_neighbors=k, leaf_size=i) #uses optimal k value so we can iterate on progress already made
    kmeans.fit(X_train, y_train)
    y_pred = kmeans.predict(X_vali) #Same as before. Test for each leaf_size limit
    accTemp = (np.sum(y_pred == y_vali)/len(y_vali))*100
    if(accTemp > accMax):
        leaf_size = i
        accMax = accTemp
    accuracy_l.append(round(accTemp,2)) #Track leaf_size accuracy

#Validation cycle (p)
accuracy_p = []
# Only looking at two types of distance measurement so no need for a loop
kmeans = KNeighborsClassifier(n_neighbors=k, leaf_size=leaf_size, p=1)
# p=1 -> Manhatten
kmeans.fit(X_train, y_train)
y_pred = kmeans.predict(X_vali)
acc1 = (np.sum(y_pred == y_vali)/len(y_vali))*100
accuracy_p.append(round(acc1, 2))
kmeans = KNeighborsClassifier(n_neighbors=k, leaf_size=leaf_size, p=2)
# p=2 -> Euclidean
kmeans.fit(X_train, y_train)
y_pred = kmeans.predict(X_vali)
acc2 = (np.sum(y_pred == y_vali)/len(y_vali))*100
accuracy_p.append(round(acc2, 2))
if (acc1 > acc2): #Find which value gave superior accuracy
    p = 1
else:
    p = 2
if (acc1 == acc2):
    print("Distance method was inconsequential") #Note if this didn't matter

#Analysis based on optimised hyperparameters
print("Fully trained. Selected hyperparameters:\nk: ", k, "\nLeaf size: ", leaf_size, "\nMeasurement distance: ", end='')
if (p == 1):
    print("Manhatten")
elif (p == 2):
    print("Euclidean")
else:
    print ("Error undf")
knn = KNeighborsClassifier(n_neighbors=k, leaf_size=leaf_size, p=p) #Assemble final optimal model
knn.fit(X_train, y_train)
toc = time.perf_counter()
time_train = round((toc-tic),4) #Stop training timer. We have the final model.
tic = time.perf_counter() #Start testing timer
y_pred = knn.predict(X_test) #Now we can test using the test set (rather than validation)
toc = time.perf_counter()
time_test = round((toc-tic),4) #Record testing time

# Results display
print("\nFully tested. Displaying results:")
# Plot accuracy results for various k values
plt.figure(figsize=(6, 6))
plt.plot(range(1, k_lim), accuracy_k, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Accuracy per K value')
plt.xlabel('K Value')
plt.ylabel('Tested Accuracy')
plt.show()
# Plot accuracy results for various leaf sizes
plt.figure(figsize=(6, 6))
plt.plot(range(1, l_lim), accuracy_l, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Accuracy for given leaf size')
plt.xlabel('Leaf size n')
plt.ylabel('Tested Accuracy')
plt.show()

# Measurement Reports
# Print out distance measurement comparison. Only two values so no need for plot
print("Distance comparison: ", "Manhatten(", accuracy_p[0], "%) | Euclidean: (", accuracy_p[1], "%)")
# SKLearn classification report
print(classification_report(y_test, y_pred))
# Time outputs
print("Data collection time: ", time_data, "s")
print("Training time: ", time_train, "s")
print("Testing time: ", time_test, "s")

# Plot a pie chart showing ratio of feature extraction time vs. training time.
# Testing time omitted after it was found to be completely negligable compared to the other two
pie_labels = 'Feature extraction', 'Training'#, 'Testing'
time_sum = time_data+time_train
pie_vals = [time_data/time_sum, time_train/time_sum]
fig1, ax1 = plt.subplots()
ax1.pie(pie_vals, labels=pie_labels, startangle=90)
ax1.axis('equal')
plt.show()

# KNN algorithm finish