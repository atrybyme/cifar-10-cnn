
# coding: utf-8

# In[1]:


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


# In[2]:


#import dataset,reshape the data,one hot encode  y, subtract the data by  mean and divide by standard deviation
with open('data_batch_1','rb') as fo:
    dict1 = pickle.load(fo,encoding='bytes')
with open('data_batch_2','rb') as fo:
    dict2 = pickle.load(fo,encoding='bytes')
with open('data_batch_3','rb') as fo:
    dict3 = pickle.load(fo,encoding='bytes')
with open('data_batch_4','rb') as fo:
    dict4 = pickle.load(fo,encoding='bytes')
with open('data_batch_5','rb') as fo:
    dict5 = pickle.load(fo,encoding='bytes')
with open('test_batch','rb') as fo:
    dict_test = pickle.load(fo,encoding='bytes')
x_train_raw = np.zeros((50000,3072))
y_train_raw = np.zeros(50000)
x_train_raw[0:10000]=dict1[b'data']
x_train_raw[10000:20000]=dict2[b'data']
x_train_raw[20000:30000]=dict3[b'data']
x_train_raw[30000:40000]=dict4[b'data']
x_train_raw[40000:50000]=dict5[b'data']
y_train_raw[0:10000] =dict1[b'labels']
y_train_raw[10000:20000] =dict2[b'labels']
y_train_raw[20000:30000] =dict3[b'labels']
y_train_raw[30000:40000] =dict4[b'labels']
y_train_raw[40000:50000] =dict5[b'labels']
x_test_raw = dict_test[b'data']
y_test = dict_test[b'labels']
x_train = np.zeros((50000,32,32,3))
x_test = np.zeros((10000,32,32,3))
y_train = np_utils.to_categorical(y_train_raw)
mean = [125.307, 122.95, 113.865]
std  = [62.9932, 62.0887, 66.7048]
x_train[:,:,:,0]=(np.reshape(x_train_raw[:,0:1024],(50000,32,32))- mean[0])/std[0]
x_train[:,:,:,1]=(np.reshape(x_train_raw[:,1024:2048],(50000,32,32))-mean[1])/std[1]
x_train[:,:,:,2]=(np.reshape(x_train_raw[:,2048:3072],(50000,32,32))-mean[2])/std[2]
x_test[:,:,:,0]=(np.reshape(x_test_raw[:,0:1024],(10000,32,32))-mean[0])/std[0]
x_test[:,:,:,1]=(np.reshape(x_test_raw[:,1024:2048],(10000,32,32))-mean[1])/std[1]
x_test[:,:,:,2]=(np.reshape(x_test_raw[:,2048:3072],(10000,32,32))-mean[2])/std[2]


# In[5]:


#build vgg_16 model(the model does not uses the exact architecture of vgg16 stated in the published paper)
#this is due to lack of computational power and time
def vgg_model():
    ##1st socket
    model = Sequential()
    model.add(Conv2D(64,(3,3),padding='same',input_shape=(32,32,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    ##2nd socket
    model.add(Conv2D(128, (3, 3),padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    #3rd socket
    model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    #4th socket
    model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #sth socket
    model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2048,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1028,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


# In[7]:


model = vgg_model()
start_time = time.clock()
model.fit(x_train,y_train,batch_size=128,epochs=50,verbose=1,validation_data=(x_test,np_utils.to_categorical(y_test)))
end_time = time.clock()
#calculate the time elapsed in the training
print("Time elapsed in training: ",(end_time-start_time)/60.00 ,"minutes")

#Calculate the training set accuracy over the whole data
l = model.predict(x_train,verbose=1,batch_size=100)
train_prediction = np.argmax(l,axis=1)
e= (np.equal(np.argmax(y_train,axis=1),train_prediction))*1
print("ACCURACY ON Training sET",(np.sum(e))/len(y_train))

#check the testing accuracy
l = model.predict(x_test,verbose=1)
prediction_test= np.argmax(l,axis=1)
e= (np.equal(y_test,prediction_test))*1
print("ACCURACY ON Testing sET",(np.sum(e))/len(y_test))


# In[8]:


#Additional Information about the model
print(model.summary())


# In[9]:


#Save the model
model.save('vgg16_model.h5')

