# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 13:15:24 2020

@author: AjaySai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
tf.__version__

"""## Importing the dataset"""

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values

"""## Feature Scaling"""

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

"""##Training the SOM"""

from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len= 15, sigma= 1.0, learning_rate = 0.5,random_seed=19)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

"""##Visualizing the results"""

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

"""## Finding the frauds"""

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(4,5)], mappings[(3,7)]), axis = 0)
frauds = sc.inverse_transform(frauds)

"""##Printing the Fraunch Clients"""

print('Fraud Customer IDs')
for i in frauds[:, 0]:
  print(int(i))
  
  
  
"""#Part 2 - Going from Unsupervised to Supervised Deep Learning

##Create Matrix of Features
"""

customers = dataset.iloc[:, :-1].values

"""## Create Dependent Variable"""

is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
  if dataset.iloc[i,0] in frauds:
    is_fraud[i] = 1

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(customers, is_fraud, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Training the k Nearest neignbours model on the Training set
from sklearn.neighbors import KNeighborsClassifier
knclass = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knclass.fit(X_train, y_train)

#Predicting Results
y_pred = knclass.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix, accuracy_score,precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
acs=accuracy_score(y_test, y_pred)
prs=precision_score(y_test, y_pred)
res=recall_score(y_test, y_pred)
print("acuuracy =",acs,"precision =",prs,"Sensitivity =",res)

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_pred)
print('roc_auc_score for K-NN model: ', roc_auc_score(y_test, y_pred))
plt.title('Receiver Operating Characteristic - K-NN model')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()