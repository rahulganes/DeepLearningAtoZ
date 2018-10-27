# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 01:18:51 2018

@author: Rahul
"""

#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential #the deep learning that is going to be used
from keras.layers import Dense #for adding hidden layers
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split

#importing the dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, 3:13].values #features provided by the bank on which the prediction is to be based on
Y = dataset.iloc[:, 13].values #whether the customer stays or leaves

#The Problem is to find the customers who would churn(leave the bank)

#Encoding the categorical data
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1]) #as encoding should follow no hierarchy
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #to avoid dummy variable trap

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)


#splitting the dataset into Train and Test dataset
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

#Building a Artificial Neural Network
model = Sequential()

#adding the first hidden layer(input) of the network
model.add(Dense(output_dim = 6,init='uniform',activation='relu',input_dim = 11))
#output_dim --> dimensions of the output from the layer & it is based on our choice,there is no rule for fixing values of it
#init --> intializing all the weights as uniform
#activation function --> rectifier is used
#input_dim -->  it is mandatory to mention the input's dimensions for the first hidden layer

#adding the second layer of the network
model.add(Dense(output_dim = 6,init = 'uniform',activation='relu'))

#adding the output layer of the network
model.add(Dense(output_dim = 1,init='uniform',activation='sigmoid'))
#it was recommended to have the activation function of final layer as sigmoid

#compiling the Artificial Neural Network
model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
#optimizer--> to chose optimum for the model eg.,adam(a type of Stochastic gradient descent)
#loss fuction --> if more than 2 category, use categorical_crossentropy
#metrics-->accuracy criterion is used to evaluate and improve the model

#training the model
model.fit(X_train, Y_train, batch_size = 10, epochs = 100)
#batch training is used

# Predicting the Test set results
Y_pred = model.predict(X_test)
Y_pred = (Y_pred > 0.5) 

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

model.save_weights('weights.h5')
#saving the model for later use or deployment

country = labelencoder_X_1.transform(['France'])[0]
credit = 600
gender = labelencoder_X_2.transform(['Male'])[0]
age = 40
tenure = 3
bal = 60000
nop = 2
card = 1
active = 1
sal = 50000

val = []

val.append(credit)
val.append(country)
val.append(gender)
val.append(age)
val.append(tenure)
val.append(bal)
val.append(nop)
val.append(card)
val.append(active)
val.append(sal)

#predicting for a single value

#forming a single vector
val = np.array(val)

#encoding and scaling a single vector
val = onehotencoder.transform(val.reshape(1,-1)).toarray()[:, 1:]
val = sc.transform(val)

new_prediction = model.predict(val)
new_prediction = (new_prediction>0.5)

