# -*- coding: utf-8 -*-
"""
Created on Fri Nov 4 23:45:17 2022

@author: Hamid
"""

## Averaged
import numpy as np
import pandas as pd
from numpy import linalg
import matplotlib.pyplot as plt

# Hyperparams
epochs = 10
r      = 1
w_init = np.zeros((1 , df_train.shape[1]-1))

# Define Perceptron
def perceptron_avg(df , w , epochs , r):
    W = np.zeros((1 , df_train.shape[1]-1))
    for e in range(0 , epochs):
        df_shuffle = df.sample(frac=1).reset_index(drop=True)
        for r in range(df_shuffle.shape[0]):
            yi = df_shuffle.iloc[r,-1]
            xi = df_shuffle.iloc[r,:-1].to_numpy()
            if ( (2*yi-1)*np.dot(w,xi) )<=0:
                w=w+(r*(2*yi-1)*xi.transpose())
            W=W+w
    return W

# Prediction fn
def predict(df , w_avg):
    e = 0
    for r in range(0 , df.shape[0]):
        x_i= df.iloc[r,:-1]
        p  = np.sign(np.dot(w_avg,x_i))
        if p==0:
            p=1
        if p!=2*(df.iloc[r,-1])-1:
            e += 1
    e_avg = e/df.shape[0]
    return e_avg

col = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

df_train = pd.read_csv('train.csv' , names=columns , dtype=np.float64())
df_test  = pd.read_csv('test.csv'  , names=columns , dtype=np.float64())
df_train.insert(0, 'b', 1)
df_test. insert(0, 'b', 1)

w_avg  = perceptron_avg(df_train, w_init, epochs, r)
e_test = predict(df_test, w_avg)

print('weight avg: ', w_avg)
print('test error: '    , e_test)