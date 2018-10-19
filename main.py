# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:08:20 2018
@author: rg
"""

#Let's import the libraries necessary for building a RNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout

#getting the trainig dataset
dataset = pd.read_csv('Google_Stock_Price_Train.csv')
train = dataset.iloc[:,1:2].values

#Feature Scaling of dataset
sc = MinMaxScaler(feature_range=(0,1))
train_data = sc.fit_transform(train) 

#creating a datastructure with 60 timesteps and 1 output
X_train = []
Y_train = []
for i in range(60,1258):#for each i/p ,60 prev i/p is considered
    X_train.append(train_data[i-60:i, 0])
    Y_train.append(train_data[i, 0])
X_train,Y_train = np.array(X_train),np.array(Y_train)

#Reshaping Data
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1], 1))

#intialising our Regressor model
model = Sequential()

#Adding the  LSTM layers and adding dropout(to avoid overfitting)
model.add(LSTM(units = 50,return_sequences = True, input_shape = (X_train.shape[1],1)))
model.add(Dropout(0.2))#first
model.add(LSTM(units = 50,return_sequences = True))
model.add(Dropout(0.2))#second
model.add(LSTM(units = 50,return_sequences = True))
model.add(Dropout(0.2))#third
model.add(LSTM(units = 50))
model.add(Dropout(0.2))#fourth

#Adding the final output layer
model.add(Dense(units = 1))

#compilling RNN
model.compile(optimizer='adam',loss = 'mean_squared_error')

#Fitting the RNN to train data set
model.fit(X_train , Y_train,epochs = 100, batch_size = 32)

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
test = dataset_test.iloc[:,1:2].values

dataset_total = pd.concat((train['open'],test['open']), axis = 0)




