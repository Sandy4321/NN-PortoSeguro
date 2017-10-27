# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:14:59 2017

@author: andersonduran
"""
import pandas as pd
import numpy as np

#IMPORTING DATASET
dataframe = pd.read_csv('../../data/train.csv')

#DATAFRAME SHAPE
dataframe.shape

#DATA TYPES INSIDE DATAFRAME
from collections import Counter
Counter(dataframe.dtypes.values)

#DATAFRAME CLASSES
dataframe['target'].unique()

#ARE THERE NULL VALUES ON DATASET?
dataframe.isnull().any().any()

#NULL WERE REPLACED BY -1, CHECKING AGAIN FOR NULL VALUES
df_nulls = dataframe.replace(-1, np.NaN)
df_nulls.isnull().any().any()

#COUNTING HOW MANY NULLS INSIDE THE FEATURES
null_feats = df_nulls.isnull().sum()

#GROUPING CLASSES TO CHECK LABELS DISTRIBUTION
import math
thist = dataframe.groupby(['target'], as_index=False).count()['id']
math.floor(thist[0]/thist[1])

#RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=15, max_depth=8, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
rf.fit(dataframe.drop(['id', 'target'],axis=1), dataframe.target)
features = dataframe.drop(['id', 'target'],axis=1).columns.values

#PLOTING CHARTS
import matplotlib.pyplot as plt

#CLASSES DISTRIBUTION
x_axis = np.arange(2)
fig, ax = plt.subplots()
ax.set_ylabel('Labels Distribution')
ax.set_title('Scores by Labels')
plt.bar(x_axis, thist)
plt.xticks(x_axis, ('Negative', 'Positive'))
plt.show()

#FEATURE IMPORTANCE SCATTER
y = rf.feature_importances_
x = np.arange(len(y))
locs, labels = plt.xticks(x, features)
plt.setp(labels, rotation=45)
plt.title('Features Importances')
plt.scatter(x, y, s=20, alpha=0.5)
plt.show()

#FEATURE IMPORTANCE BAR PLOT
y, features = (list(x) for x in zip(*sorted(zip(y, features), reverse=True)))
fig, ax = plt.subplots()
y_pos = np.arange(len(y))
ax.barh(y_pos, y, align='center')
ax.invert_yaxis()
ax.set_xlabel('Importances')
ax.set_ylabel('Features')
ax.set_yticks(x)
ax.set_yticklabels(features)
ax.set_title('Features Importances')
plt.show()

#PLOT BINARY FEATURES DISTRIBUTION
bin_col = [col for col in dataframe.columns if '_bin' in col]
zero_list = []
one_list = []
for col in bin_col:
	zero_list.append((dataframe[col]==0).sum())
	one_list.append((dataframe[col]==1).sum())

ind = np.arange(len(bin_col))
width = 0.5
p1 = plt.bar(ind, zero_list, width, color='#d62728')
p2 = plt.bar(ind, one_list, width, bottom=zero_list)
plt.ylabel('Scores')
plt.title('Value distribution inside binary features')
plt.xticks(ind, bin_col)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.legend((p1[0], p2[0]), ('Negative', 'Positive'))
plt.show()

#PLOT FEATURES CORRELATION
import seaborn as sns
df_float = dataframe.select_dtypes(include=['float64'])
colormap = plt.cm.afmhot
plt.figure(figsize=(16,12))
plt.title('Pearson correlation of continuous features', y=1.05, size=15)
sns.heatmap(df_float.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()
df_int = dataframe.select_dtypes(include=['int64'])

#PLOT FEATURES HISTOGRAM
feature_values = list(dataframe["ps_reg_01"])
n, bins, patches = plt.hist(feature_values, normed=1, alpha=0.75)
plt.title('Histogram for feature ' + dataframe["ps_reg_01"])
plt.grid(True)
plt.show()