# -*- coding: utf-8 -*-
"""
Created on Sat Nov 5 10:06:39 2022

@author: Hamid
"""

## Standard
import pandas as pd
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

# Hyperparams
epochs = 10
r      = 1
w_init = np.zeros((1,df_train.shape[1]-1))

# Define the Perceptron
def perceptron_std(df, w, epochs, r):
    for e in range(epochs):
        df_shuffle = df.sample(frac=1).reset_index(drop=True)
        for r in range(df_shuffle.shape[0]):
            y_i = df_shuffle.iloc[r,-1]
            x_i = df_shuffle.iloc[r,:-1].to_numpy()
            if ((2*y_i-1)*(np.dot(w,x_i))) <= 0:
                w += (r*(2*y_i-1)*(x_i.transpose()))
    return w            

# Prediction fn
def predict(df, w):
    e = 0
    for r in range(df.shape[0]):
        x_i = df.iloc[r,:-1]        
        p = np.sign(np.dot(w,x_i))
        if p==0:
            p=1
        if p!=2*(df.iloc[r,-1])-1:
            e += 1
    e_avg = e/df.shape[0]
    return e_avg

columns=['variance','skewness','curtosis','entropy','label']

df_train = pd.read_csv('train.csv', names=columns, dtype=np.float64())
df_test  = pd.read_csv('test.csv' , names=columns, dtype=np.float64())
df_train.insert(0, 'b', 1)
df_test .insert(0, 'b', 1)

w_final = perceptron_std(df_train, w_init, epochs, r)
e_test = predict(df_test, w_final)

print('w final: ', w_final)
print('error: ' , e_test)