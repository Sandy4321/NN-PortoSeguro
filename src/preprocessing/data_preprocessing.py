# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 00:17:59 2017

@author: andersonduran
"""
import pandas as pd
import numpy as np

#LOAD DATASET
df_train = pd.read_csv('../../data/train.csv')
df_test = pd.read_csv('../../data/test.csv')

#REPLACE -1s FOR NUMPY NaN
df_train = df_train.replace(-1, np.NaN)
df_test = df_test.replace(-1, np.NaN)

#DROP UNDESIRED FEATURES
df_train.drop('id', axis=1, inplace=True)
df_train.drop('ps_ind_10_bin', axis=1, inplace=True)
df_train.drop('ps_ind_11_bin', axis=1, inplace=True)
df_train.drop('ps_ind_12_bin', axis=1, inplace=True)
df_train.drop('ps_ind_13_bin', axis=1, inplace=True)
df_train.drop('ps_car_03_cat', axis=1, inplace=True)
df_train.drop('ps_car_05_cat', axis=1, inplace=True)

df_test.drop('id', axis=1, inplace=True)
df_test.drop('ps_ind_10_bin', axis=1, inplace=True)
df_test.drop('ps_ind_11_bin', axis=1, inplace=True)
df_test.drop('ps_ind_12_bin', axis=1, inplace=True)
df_test.drop('ps_ind_13_bin', axis=1, inplace=True)
df_test.drop('ps_car_03_cat', axis=1, inplace=True)
df_test.drop('ps_car_05_cat', axis=1, inplace=True)

cat_features = []
cat_features.append('ps_reg_03')
cat_features.append('ps_car_11')
cat_features.append('ps_car_12')
cat_features.append('ps_car_14')

df_train.isnull().any().any()
df_test.isnull().any().any()

#TAKE CARE OF MISSING VALUES
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(df_train[cat_features])
df_train[cat_features] = imputer.transform(df_train[cat_features])

imputer = imputer.fit(df_test[cat_features])
df_test[cat_features] = imputer.transform(df_test[cat_features])

bin_features = []
bin_features.append('ps_ind_02_cat')
bin_features.append('ps_ind_04_cat')
bin_features.append('ps_ind_05_cat')
bin_features.append('ps_car_01_cat')
bin_features.append('ps_car_02_cat')
bin_features.append('ps_car_07_cat')
bin_features.append('ps_car_09_cat')

imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imputer = imputer.fit(df_train[bin_features])
df_train[bin_features] = imputer.transform(df_train[bin_features])

imputer = imputer.fit(df_test[bin_features])
df_test[bin_features] = imputer.transform(df_test[bin_features])

#CHECKING AGAIN FOR NULL VALUES
df_train.isnull().any().any()
df_test.isnull().any().any()

#UNDERSAMPLING TRAIN SET
zeros = df_train[df_train['target']==0].sample(n=25000)
ones = df_train[df_train['target']==1]
us_df_train = pd.concat([zeros,ones], axis=0)

#CHECK TRAIN SET
import math
classdist = us_df_train.groupby(['target'], as_index=False).count()['ps_ind_01']
print('Class value = 0 --> %s examples.' % (classdist[0]))
print('Class value = 1 --> %s examples.' % (classdist[1]))
print('Classes ratio is %d:1.' % (math.floor(classdist[0]/classdist[1])))

Xt = us_df_train.iloc[:, 1:52].values
yt = us_df_train.iloc[:, 0].values

#THE CHALENGE TEST SET WHOSE VALUES ARE NOT KNOWN
X_chalenge = df_test.iloc[:, 0:51].values

#SPLITING DATASET INTO TRAIN, DEV AND TEST SET
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xt, yt, test_size=0.2)
X_train, X_dev, y_train, y_dev = train_test_split(Xt, yt, test_size=0.2)

#DATASET SCALING AND NORMALIZATION STEP
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_dev = sc.transform(X_dev)
X_test = sc.transform(X_test)
X_chalenge = sc.transform(X_chalenge)

#SAVE IT TO THE H5 FILE
import h5py
h5file = h5py.File('../model/dataset.h5', 'w')

h5file.create_dataset('X_train', (len(X_train), X_train[0].shape[0]), np.float32)
h5file.create_dataset('X_dev', (len(X_dev), X_dev[0].shape[0]), np.float32)
h5file.create_dataset('X_test', (len(X_test), X_test[0].shape[0]), np.float32)

h5file.create_dataset('y_train', (len(y_train), 1), np.int8)
h5file.create_dataset('y_dev', (len(y_dev), 1), np.int8)
h5file.create_dataset('y_test', (len(y_test), 1), np.int8)

for i in range(len(X_train)):
    h5file['X_train'][i, ...] = X_train[i]
for i in range(len(X_dev)):
    h5file['X_dev'][i, ...] = X_dev[i]
for i in range(len(X_test)):
    h5file['X_test'][i, ...] = X_test[i]
for i in range(len(y_train)):
    h5file['y_train'][i, ...] = y_train[i]
for i in range(len(y_dev)):
    h5file['y_dev'][i, ...] = y_dev[i]
for i in range(len(y_test)):
    h5file['y_test'][i, ...] = y_test[i]

h5file.close()

h5file2 = h5py.File('../model/chalenge.h5', 'w')
h5file2.create_dataset('X_chalenge', (len(X_chalenge), X_chalenge[0].shape[0]), np.float32)

for i in range(len(X_chalenge)):
    h5file2['X_chalenge'][i, ...] = X_chalenge[i]

h5file2.close()








