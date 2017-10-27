# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:14:59 2017

@author: andersonduran
"""
from model_impl import define_predict
from model_impl import define_model

#IMPORTING THE DATASET
import h5py
import numpy as np

#LOADING DATASET AND TRANSPOSING TO VECTORIZED IMPLEMENTATION
h5file  = h5py.File('dataset.h5', 'r')
X_train = np.array(h5file['X_train'], dtype=np.float32).T
X_dev   = np.array(h5file['X_dev'], dtype=np.float32).T
X_test  = np.array(h5file['X_test'], dtype=np.float32).T
y_train = np.array(h5file['y_train']).T
y_dev   = np.array(h5file['y_dev']).T
y_test  = np.array(h5file['y_test']).T

#DEFINING MODEL ARCHITECTURE
features = X_train.shape[0]
layers = [features, 20, 2, 1]

#DEFINING MODEL
train = define_model(layers)

#NUMBER OF ITERATIONS OVER THE DATASET
epochs = 10000

#TRAINING NETWORK
cost = W1 = W2 = W3 = b1 = b2 = b3 = None
for i in range(epochs):
    cost, W1, W2, W3, b1, b2, b3 = train(X_train, y_train, 0.1)
    if i % 100 == 0:
        print('Cost on epoch %i: %s' %(i, cost))

predictor = define_predict(W1, b1, W2, b2, W3, b3)

#PREDICTING THE OUTPUT
y_train_hat = predictor(X_train)
y_dev_hat = predictor(X_dev)
y_test_hat = predictor(X_test)

#CONVERTING TO BINARY OUTPUT
from utils import to_binary
y_train_hat = to_binary(y_train_hat)
y_dev_hat   = to_binary(y_dev_hat)
y_test_hat  = to_binary(y_test_hat)

#CHECKING ACCURACY
from sklearn.metrics import accuracy_score
accuracy_score(y_train[0], y_train_hat[0], normalize=True)

#CHECKING RECALL
from sklearn.metrics import recall_score
recall_score(y_train[0], y_train_hat[0])

#CHECKING F1 SCORE
from sklearn.metrics import f1_score
f1_score(y_train[0], y_train_hat[0])

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train[0], y_train_hat[0])


def to_binary(y):
    pred = np.zeros((1, y.shape[1]))
    for i in range(y.shape[1]):
        if y[0, i] > 0.5:
            pred[0, i] = 1
    return pred
