# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 07:38:05 2022


"""
import os
import glob
import tensorflow as tf
import numpy
import numpy
import cv2
from PIL import Image
import keras
from tensorflow.keras.models import Sequential
from keras.layers import Dense,Conv1D, Conv2D,MaxPool1D, MaxPool2D,Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from matplotlib import pyplot as plt
import scipy
from skimage.measure import block_reduce
import scipy.io as sio
import numpy
import hdf5storage

import scipy.io as sio
import numpy
import h5py

test = sio.loadmat('LB.mat')

a=test['LB'][0,:]
N=a.shape[0]
y=numpy.zeros(a.shape[0])
for i in range(0,N):
    y[i]=float(a[i])
    


w = hdf5storage.loadmat('X.mat')



numpy.random.seed(1)

FOLDER_NUM=1
IMG_NUM=20000
N=FOLDER_NUM*IMG_NUM*2

Sx=128
Sy=128
data=[]
lable=[]
Data=w['X']
x=numpy.zeros((N,Sy,Sx,1))

C=0
# PD
for i in range(0,N):

    x[i,:,:,0]=Data[:,:,0,i]

   
    
# shuffle data
IND= list(range(x.shape[0]))
numpy.random.shuffle(IND)

X2=numpy.zeros((x.shape[0],x.shape[1],x.shape[2],x.shape[3]))
X2[:,:,:,:]=x[IND,:,:,:]
x[:,:,:,:]=X2[:,:,:,:]
y2=numpy.zeros(N)
y2[:]=y[IND]
y[:]=y2[:]

EPOCH=10
epoch_range=range(EPOCH)

K=5
Ntr=int(N*(K-1)/(K))
train=numpy.zeros((K,Ntr),'int')
test=numpy.zeros((K,N-Ntr),'int')

for D in range(0,K):
    for i in range(0,N-Ntr):
        test[D,i]=i+D*(N-Ntr)

for D in range(0,K):
    C=0
    for i in range(0,N):
        if(i>=D*(N-Ntr) and i<(D+1)*(N-Ntr)):
            continue
        
        train[D,C]=i
        C=C+1


TP=numpy.zeros(K)
TN=numpy.zeros(K)
FN=numpy.zeros(K)
FP=numpy.zeros(K)

Result_M={}
for FD in range(0,K):
    
    modelF=Sequential()
    modelF.add(Conv2D(24,3,padding="same",activation="relu",input_shape=(x.shape[1],x.shape[2],x.shape[3])))
    modelF.add(MaxPool2D(pool_size=(2,2)))

    modelF.add(Conv2D(32,kernel_size=(3,3),padding="same",activation="relu"))
    modelF.add(MaxPool2D(pool_size=(2,2)))

    modelF.add(Conv2D(16,kernel_size=(3,3),padding="same",activation="relu"))
    modelF.add(MaxPool2D(pool_size=(2,2)))

    modelF.add(Flatten())
    modelF.add(layer=Dropout(0.2))
    modelF.add(Dense(64,activation="relu"))
    modelF.add(Dense(2,activation="softmax"))
    modelF.summary()
    opt=Adam(lr=0.001)
    modelF.compile(optimizer=opt,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

    M4=modelF.fit(x[train[FD,:],:,:],y[train[FD,:]],validation_split=0.1,epochs=EPOCH)
    
    Result_M[FD]=M4

      
    
    
    
    
    
    
    print('tr')    
    # calculate output of each model

  
    
    # Predict

    E=modelF.predict(x[test[FD,:],:,:])

    for i in range(0,len(test[FD,:])):
        if(E[i,0] < E[i,1]):
            # index = 1
            if(int(y[test[FD,i]])==1):
                TP[FD]=TP[FD]+1
            else:
                FN[FD]=FN[FD]+1
            
        else:
            # index = 0
            if(int(y[test[FD,i]])==0):
                TN[FD]=TN[FD]+1
            else:
                FP[FD]=FP[FD]+1
            
            
print('TN =',TN)
print('TP =',TP)
print('FN =',FN)
print('FP =',FP)

for i in range(0,K):
    accuracy=(TN[i]+TP[i])/(TN[i]+TP[i]+FN[i]+FP[i])

    if(TP[i]+FP[i]==0):
        precision=0
    else:
        precision=(TP[i])/(TP[i]+FP[i])
    DSC=(2*TP[i])/(2*TP[i]+FN[i]+FP[i])

    print('[',i,'] accuracy = ',accuracy)

    print('[',i,'] precision = ',precision)

    print('[',i,'] DSC = ',DSC)
    

TP_A=numpy.sum(TP)
TN_A=numpy.sum(TN)
FP_A=numpy.sum(FP)
FN_A=numpy.sum(FN)

accuracy=(TN_A+TP_A)/(TN_A+TP_A+FN_A+FP_A)

precision=(TP_A)/(TP_A+FP_A)
DSC=(2*TP_A)/(2*TP_A+FN_A+FP_A)

rc=TP_A/(TP_A+FN_A)

print('TOTAL ')
print('accuracy = ',accuracy)

print('precision = ',precision)

print('Fscore = ',DSC)

print('recall = ',rc)




plt.show()
plt.subplot(1,1,1)
for i in range(0,K):


    acc4=Result_M[i].history['accuracy']
    val_accuracy4=Result_M[i].history['val_accuracy']
  

  
    plt.plot(epoch_range,acc4,label='accuracy')
    plt.plot(epoch_range,val_accuracy4,label='accuracy')
  
    #plt.plot(epoch_range,val_accuracy4,label='accuracy')
  
plt.title('2D accuracy')
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.legend(['train 1','val 1','train 2','val 2','train 3','val 3','train 4','val 4','train 5','val 5'])
plt.savefig('CNN-Acc.png', dpi=1200)
plt.show()
  
  

plt.subplot(1,1,1)
for i in range(0,K):
    
    loss4=Result_M[i].history['loss']
    val_loss4=Result_M[i].history['val_loss']


    plt.plot(epoch_range,loss4,label='loss')
    plt.plot(epoch_range,val_loss4,label='loss')
  
    #plt.plot(epoch_range,val_accuracy4,label='accuracy')
  
plt.title('loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(['train 1','val 1','train 2','val 2','train 3','val 3','train 4','val 4','train 5','val 5'])
plt.savefig('CNN-loss.png', dpi=1200)
plt.show()
