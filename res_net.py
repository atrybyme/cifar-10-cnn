# Author : Shubhansh Awasthi
# coding: utf-8

# In[6]:


#importing required libraries
import pickle
import numpy as np
from keras.models import Model
from keras.layers import Dense,Input,add,Flatten,Dropout,Activation
from keras.layers import GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
import time


# In[2]:


#loading the data, subtact the mean and divide by standard deviation
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

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
mean = [125.307, 122.95, 113.865]
std  = [62.9932, 62.0887, 66.7048]
x_train[:,:,:,0]=(np.reshape(x_train_raw[:,0:1024],(50000,32,32))- mean[0])/std[0]
x_train[:,:,:,1]=(np.reshape(x_train_raw[:,1024:2048],(50000,32,32))-mean[1])/std[1]
x_train[:,:,:,2]=(np.reshape(x_train_raw[:,2048:3072],(50000,32,32))-mean[2])/std[2]
x_test[:,:,:,0]=(np.reshape(x_test_raw[:,0:1024],(10000,32,32))-mean[0])/std[0]
x_test[:,:,:,1]=(np.reshape(x_test_raw[:,1024:2048],(10000,32,32))-mean[1])/std[1]
x_test[:,:,:,2]=(np.reshape(x_test_raw[:,2048:3072],(10000,32,32))-mean[2])/std[2]


# In[3]:


#to block 1 for same , and other for increment
def residual_block_same(input_data,output_channel):
    norm_lyr_0 = BatchNormalization()(input_data)
    relu_lyr_0 = Activation('relu')(norm_lyr_0)
    conv_1 = Conv2D(output_channel,(3,3),padding='same')(relu_lyr_0)
    norm_lyr_1 = BatchNormalization()(conv_1)
    relu_lyr_1 = Activation('relu')(norm_lyr_1)
    conv_2 = Conv2D(output_channel,(3,3),padding='same')(relu_lyr_1)
    block = add([conv_2,input_data])
    return block
def residual_block_increase(input_data,output_channel):
    projection = Conv2D(output_channel,(1,1),strides=(2,2),padding='same')(input_data)
    norm_lyr_0 = BatchNormalization()(input_data)
    relu_lyr_0 = Activation('relu')(norm_lyr_0)
    conv_1 = Conv2D(output_channel,(3,3),strides=(2,2),padding='same')(relu_lyr_0)
    norm_lyr_1 = BatchNormalization()(conv_1)
    relu_lyr_1 = Activation('relu')(norm_lyr_1)
    conv_2 = Conv2D(output_channel,(3,3),padding='same')(relu_lyr_1)
    block = add([conv_2,projection])
    return block


# In[13]:


#applying model
input_details = Input(shape=(32,32,3))
x=Conv2D(16,(3,3),padding='same')(input_details)
#x = Dropout(0.4)(x)
for _ in range(5):
    x=residual_block_same(x,16)
x = residual_block_increase(x,32)
for _ in range(4):
    x=residual_block_same(x,32)
x = residual_block_increase(x,64)

for _ in range(4):
    x=residual_block_same(x,64)

x = BatchNormalization()(x)
x = Activation('relu')(x)
#x = Dropout(0.50)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(10,activation='softmax')(x)
full_model = Model(input_details,x)
full_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])





# In[14]:


start_time = time.clock()
full_model.fit(x_train,y_train,batch_size=128,epochs=50,verbose=1,validation_data=(x_test,np_utils.to_categorical(y_test)))
end_time = time.clock()
#calculate the time elapsed in the training
print("Time elapsed in training: ",(end_time-start_time)/60.00 ,"minutes")





# In[15]:


#Calculate the training set accuracy over the whole data
l = full_model.predict(x_train,verbose=1,batch_size=100)
train_prediction = np.argmax(l,axis=1)
e= (np.equal(np.argmax(y_train,axis=1),train_prediction))*1
print("ACCURACY ON Training sET",(np.sum(e))/len(y_train))


# In[16]:


#check the testing accuracy
l = full_model.predict(x_test,verbose=1)
prediction_test= np.argmax(l,axis=1)
e= (np.equal(y_test,prediction_test))*1
print("ACCURACY ON Testing sET",(np.sum(e))/len(y_test))


# In[18]:


#Additional Information about the model
print(full_model.summary())


# In[19]:


#Save the model
full_model.save('resnet_model.h5')

# Author : Shubhansh Awasthi
