# Author : Shubhansh Awasthi
# coding: utf-8

# In[3]:


#importing required libraries
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
import time
from keras.models import load_model


# In[4]:


#import dataset,reshape the data,, subtract the data by  mean and divide by standard deviation
with open('test_batch','rb') as fo:
    dict_test = pickle.load(fo,encoding='bytes')
x_test_raw = dict_test[b'data']
y_test = dict_test[b'labels']
x_test = np.zeros((10000,32,32,3))
mean = [125.307, 122.95, 113.865]
std  = [62.9932, 62.0887, 66.7048]
x_test[:,:,:,0]=(np.reshape(x_test_raw[:,0:1024],(10000,32,32))-mean[0])/std[0]
x_test[:,:,:,1]=(np.reshape(x_test_raw[:,1024:2048],(10000,32,32))-mean[1])/std[1]
x_test[:,:,:,2]=(np.reshape(x_test_raw[:,2048:3072],(10000,32,32))-mean[2])/std[2]


# In[8]:


model_name='net_in_net_model.h5'
model =load_model(model_name)


# In[7]:




#check the testing accuracy
l = model.predict(x_test,verbose=1)
prediction_test= np.argmax(l,axis=1)
e= (np.equal(y_test,prediction_test))*1
print("ACCURACY ON Testing sET",(np.sum(e))/len(y_test))


# In[8]:


#Additional Information about the model
print(model.summary())
# Author : Shubhansh Awasthi
