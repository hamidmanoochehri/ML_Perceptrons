# -*- coding: utf-8 -*-
"""
Created on Sat Nov 5 12:03:21 2022

@author: Hamid
"""

## Voted
import pandas as pd
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

# Hyperparams
epochs = 10
r      = 1
w_init = np.zeros((1,df_train.shape[1]-1))

# Define the Perceptron
def perceptron_vot(df, w, epochs, r):
    Cm  = 0
    Cms = []
    ws  = []
    for e in range(epochs):
        df_shuffle = df.sample(frac=1).reset_index(drop=True)
        #df_shuffle = df_shuffle.reset_index(drop=True)
        for r in range(df_shuffle.shape[0]):
            y_i = df_shuffle.iloc[r,-1]
            x_i = df_shuffle.iloc[r,:-1].to_numpy()
            if (2*y_i-1)*w.dot(x_i) <= 0:
                Cms.append(Cm)
                ws.append(w)
                w += ( r*(2*y_i-1)*(x_i.transpose()) )
                Cm = 1
            else:
                Cm += 1
    Cms.append(Cm)
    ws.append(w)
    return Cms , ws

# Prediction fn
def predict(df, Cms, ws):
    e=0
    for r in range(df.shape[0]):
        pred = 0
        x_i  = df.iloc[r,:-1]
        for i in range (len(ws)):
            w = ws[i]
            p=np.sign(np.dot(w,x_i))
            if p==0:
                p=1
            p = p*Cms[i]
            pred += p

        pred = np.sign(pred)
        if pred==0:
            pred = 1    
        if pred != (2*(df.iloc[r,-1])-1):
            e += 1
    e_avg = e/df.shape[0]
    return e_avg

columns=['variance','skewness','curtosis','entropy','label']

df_train = pd.read_csv('train.csv', names=columns, dtype=np.float64())
df_test  = pd.read_csv('test.csv' , names=columns, dtype=np.float64())
df_train.insert(0, 'b', 1)
df_test .insert(0, 'b', 1)

Cms = []
ws  = []
Cms, ws = perceptron_vot(df_train, w_init, epochs, r)
print ('w\'s: '  , ws)
print ('Cm\'s: ' , Cms)
print ('number of w\'s:', len(Cms))

e_test = predict(df_test, Cms , ws)
print('test error: ', e_test)