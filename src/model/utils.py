# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 17:50:19 2017

@author: andersonduran
"""
import numpy as np

def to_binary(y):
    pred = np.zeros((1, y.shape[1]))
    for i in range(y.shape[1]):
        if y[0, i] > 0.5:
            pred[0, i] = 1
    return pred
