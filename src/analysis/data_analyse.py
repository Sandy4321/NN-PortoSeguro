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