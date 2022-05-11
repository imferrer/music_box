#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import librosa
import librosa.display
import IPython.display as ipd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import pickle


# In[2]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras import models, layers
from tensorflow.keras import backend as K

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


# #### The Medley Instruments Data Set

# In[3]:


df = pd.read_csv('~/documents/data/audio/medley/csv/Medley-solos-DB_metadata.csv')


# In[4]:


df


# In[5]:


df.nunique()


# In[6]:


df['instrument'].unique()


# In[7]:


df.groupby(['instrument']).count()


# In[8]:


df.groupby(['instrument']).count().sum()


# In[9]:


df['subset'].unique()


# In[10]:


df.groupby(['subset']).count()


# Splitting Dataframe into Train, Validation and Test Dataframes

# In[11]:


df_train = df.loc[df['subset']=='training']


# ##### Read train, validation and test dataframes

# In[12]:


df_train = pd.read_csv('~/documents/data/audio/medley/csv/Medley-solos-DB_metadata_training.csv',index_col=0)


# In[13]:


df_val = pd.read_csv('~/documents/data/audio/medley/csv/Medley-solos-DB_metadata_validation.csv',index_col=0)


# In[14]:


df_test = pd.read_csv('~/documents/data/audio/medley/csv/Medley-solos-DB_metadata_test.csv',index_col=0)


# In[15]:


df_train.head()


# #### Model for 4 instruments

# In[16]:


# processing approach:
# define dataframe_subset_instrument
# load into librosa .wav to audio array (n,)
# save arrays with numpy


# In[17]:


# starting with the Train Set: 404 elements distorted guitar


# In[18]:


# begin with two instruments


# In[19]:


instruments = df_train['instrument'].unique()


# In[20]:


df_train.groupby(['instrument']).nunique()


# In[21]:


df_val.groupby(['instrument']).nunique()


# In[22]:


df_test.groupby(['instrument']).nunique()


# Understanding the data: how many songs were used per class

# In[23]:


df_train.groupby(['instrument'])['song_id'].unique()


# In[24]:


df_val.groupby(['instrument'])['song_id'].unique()


# In[25]:


df_test.groupby(['instrument'])['song_id'].unique()


# In[26]:


# smaller data frames


# In[27]:


df_train_clarinet = df_train.loc[df_train['instrument_id']==0].set_index('subset').reset_index()


# In[28]:


df_train_guitar = df_train.loc[df_train['instrument_id']==1].set_index('subset').reset_index()


# In[29]:


df_train_singer = df_train.loc[df_train['instrument_id']==2].set_index('subset').reset_index()


# In[30]:


df_train_piano = df_train.loc[df_train['instrument_id']==4].set_index('subset').reset_index()


# In[31]:


df_train_saxo = df_train.loc[df_train['instrument_id']==5].set_index('subset').reset_index()


# In[32]:


df_train_piano.shape


# In[33]:


df_train_piano.groupby(['song_id']).count()


# In[34]:


# piano split: 
# piano train, song_id 164,165,166:
238 + 189 + 68


# In[35]:


# piano val, song_id 167, 168, 169, 170:
137 + 78 + 67 + 57 


# In[36]:


df_train_clarinet.shape,df_train_guitar.shape, 


# In[37]:


df_train_singer.shape, df_train_saxo.shape


# In[38]:


# with the validation subset


# In[39]:


df_val_clarinet = df_val.loc[df_val['instrument_id']==0].set_index('subset').reset_index()


# In[40]:


df_val_guitar = df_val.loc[df_val['instrument_id']==1].set_index('subset').reset_index()


# In[41]:


df_val_singer = df_val.loc[df_val['instrument_id']==2].set_index('subset').reset_index()


# In[42]:


df_val_saxo = df_val.loc[df_val['instrument_id']==5].set_index('subset').reset_index()


# In[43]:


# with the test subset


# In[44]:


df_test_clarinet = df_test.loc[df_test['instrument_id']==0].set_index('subset').reset_index()


# In[45]:


df_test_guitar = df_test.loc[df_test['instrument_id']==1].set_index('subset').reset_index()


# In[46]:


df_test_singer = df_test.loc[df_test['instrument_id']==2].set_index('subset').reset_index()


# In[47]:


df_test_saxo = df_test.loc[df_test['instrument_id']==5].set_index('subset').reset_index()


# #### Loading signal saved in .npy numpy array files

# In[48]:


filename_pre = "Medley-solos-DB"


# In[49]:


def load_amp_file(filename,df):
    """input: filename, for example: amp_tr_clarinet
    output: list of amplitude vectors
    """
    # load amplitude vector
    # the corresponding dataframe must be at hand
    n_elements=(len(df))
    folderpath="../../../../../../documents/data/audio/medley/audio_arrays/"
    amp_list=[]
    with open (folderpath+filename, 'rb') as f:
        for n in range(n_elements):
            amp_1 = np.load(f)
            amp_list.append(amp_1)
    return amp_list


# In[50]:


amplitude_train_clarinet = load_amp_file('amp_tr_clarinet', df_train_clarinet)


# In[51]:


amplitude_train_guitar = load_amp_file('amp_tr_guitar', df_train_guitar)


# In[52]:


amplitude_train_singer = load_amp_file('amp_tr_singer', df_train_singer)


# In[53]:


amplitude_train_piano = load_amp_file('amp_tr_piano', df_train_piano)


# In[54]:


amplitude_train_saxo = load_amp_file('amp_tr_saxo', df_train_saxo)


# In[55]:


# it works!


# In[ ]:





# In[56]:


# with the val dataframes


# In[57]:


#load validation subset amplitude arrays


# In[58]:


amplitude_val_clarinet = load_amp_file('amp_val_clarinet', df_val_clarinet)


# In[59]:


amplitude_val_guitar = load_amp_file('amp_val_guitar', df_val_guitar)


# In[60]:


amplitude_val_singer = load_amp_file('amp_val_singer', df_val_singer)


# In[61]:


len(amplitude_val_singer)


# In[62]:


# piano_ working only with the train set and splitting into val and test


# In[63]:


#with the test dataframes


# In[64]:


#load test subset amplitude arrays


# In[65]:


amplitude_test_clarinet = load_amp_file('amp_test_clarinet', df_test_clarinet)


# In[66]:


amplitude_test_guitar = load_amp_file('amp_test_guitar', df_test_guitar)


# In[67]:


amplitude_test_singer = load_amp_file('amp_test_singer', df_test_singer)


# In[68]:


#benchmark


# In[69]:


len(amplitude_train_clarinet), len(amplitude_train_guitar), len(amplitude_train_singer)


# In[70]:


# correct!


# In[71]:


len(amplitude_val_clarinet), len(amplitude_val_guitar), len(amplitude_val_singer)


# In[72]:


# correct!


# In[73]:


len(amplitude_test_clarinet), len(amplitude_test_guitar), len(amplitude_test_singer)


# In[74]:


# correct!


# In[75]:


amplitude_test_clarinet[0].shape, amplitude_test_guitar[0].shape, amplitude_test_singer[0].shape


# In[76]:


# it worked!


# In[77]:


# up here: second part, load data and ML


# MEL SPECTROGRAM

# In[78]:


def amp_list_to_mel_list(amp_list):
    """input: list of signal arrays 1d
    output: list of melspectrogram arrays 2d"""
    mel_list=[]
    for amplitude_1 in amp_list:
        mel_spectrogram_1 = librosa.feature.melspectrogram(y=amplitude_1,n_mels=129)
        mel_list.append(mel_spectrogram_1)  
    return mel_list


# In[79]:


## TRAINING SUBSET


# In[80]:


mel_spectrogram_train_clarinet = amp_list_to_mel_list(amplitude_train_clarinet)


# In[81]:


mel_spectrogram_train_guitar = amp_list_to_mel_list(amplitude_train_guitar)


# In[82]:


mel_spectrogram_train_singer = amp_list_to_mel_list(amplitude_train_singer)


# In[83]:


mel_spectrogram_train_piano = amp_list_to_mel_list(amplitude_train_piano)


# In[84]:


len(mel_spectrogram_train_clarinet), len(mel_spectrogram_train_guitar),


# In[85]:


len(mel_spectrogram_train_singer), len(mel_spectrogram_train_piano)


# In[86]:


mel_spectrogram_train_clarinet[0].shape, mel_spectrogram_train_clarinet[0].shape


# In[87]:


mel_spectrogram_train_singer[0].shape, mel_spectrogram_train_piano[0].shape


# In[88]:


X_tr_clarinet = np.stack(mel_spectrogram_train_clarinet,axis=0)
X_tr_clarinet.shape


# In[89]:


X_tr_clarinet_resh = X_tr_clarinet.reshape(251,129,129,1)


# In[90]:


X_tr_clarinet_resh.shape


# In[91]:


y_tr_clarinet = np.zeros((251,1))


# In[92]:


X_tr_guitar = np.stack(mel_spectrogram_train_guitar,axis=0)
X_tr_guitar.shape


# In[93]:


X_tr_guitar_resh = X_tr_guitar.reshape(404,129,129,1)


# In[94]:


y_tr_guitar = np.ones((404,1))


# In[95]:


X_tr_singer = np.stack(mel_spectrogram_train_singer,axis=0)
X_tr_singer.shape


# In[96]:


X_tr_singer_resh = X_tr_singer.reshape(318,129,129,-1)


# In[97]:


X_tr_singer_resh.shape


# In[98]:


y_tr_singer = np.ones((318,1))*2


# In[99]:


# train_piano set: 2401 samples
# we will use 500 for X_train, 500 for validation
# the rest 1401 for test


# In[100]:


X_tr_piano = np.stack(mel_spectrogram_train_piano[0:495])
X_tr_piano.shape


# In[101]:


X_tr_piano_resh = X_tr_piano.reshape(495,129,129,1)


# In[102]:


X_tr_piano_resh.shape


# In[103]:


y_tr_piano = np.ones((495,1))*3


# In[108]:


# X_train shape (number wav files, f domain, t domain, 1)
# number wav clarinet + guitar + singer + piano:
(251 + 404 + 318 + 495, 129, 129)


# In[109]:


X_train = np.concatenate([X_tr_clarinet_resh, X_tr_guitar_resh, X_tr_singer_resh, X_tr_piano_resh])


# In[110]:


X_train.shape


# In[111]:


y_train = np.concatenate([y_tr_clarinet, y_tr_guitar, y_tr_singer,y_tr_piano])


# In[112]:


y_train.shape


# In[113]:


# One hot encode the labels:
# Keep a copy of y test as the actual number labels:
y_train_true = y_train.copy()


# In[114]:


y_train = to_categorical(y_train)


# In[115]:


y_train.shape


# In[ ]:





# In[116]:


## VALIDATION SUBSET


# In[117]:


mel_spectrogram_val_clarinet = amp_list_to_mel_list(amplitude_val_clarinet)


# In[118]:


mel_spectrogram_val_guitar = amp_list_to_mel_list(amplitude_val_guitar)


# In[119]:


mel_spectrogram_val_singer = amp_list_to_mel_list(amplitude_val_singer)


# In[120]:


mel_spectrogram_val_piano = mel_spectrogram_train_piano[495:834]


# In[121]:


len(mel_spectrogram_val_clarinet), len(mel_spectrogram_val_guitar)


# In[122]:


len(mel_spectrogram_val_singer), len(mel_spectrogram_val_piano)


# In[123]:


mel_spectrogram_train_clarinet[0].shape, mel_spectrogram_train_clarinet[0].shape


# In[124]:


mel_spectrogram_train_singer[0].shape, mel_spectrogram_val_piano[0].shape


# In[125]:


# X_val, y val


# In[126]:


# clarinet


# In[127]:


X_val_clarinet = np.stack(mel_spectrogram_val_clarinet,axis=0)
X_val_clarinet.shape


# In[128]:


X_val_clarinet_resh = X_val_clarinet.reshape(328,129,129,1)


# In[129]:


X_val_clarinet_resh.shape


# In[130]:


y_val_clarinet = np.zeros((328,1))


# In[131]:


# guitar


# In[132]:


X_val_guitar = np.stack(mel_spectrogram_val_guitar,axis=0)
X_val_guitar.shape


# In[133]:


X_val_guitar_resh = X_val_guitar.reshape(495,129,129,1)


# In[134]:


X_val_guitar_resh.shape


# In[135]:


y_val_guitar = np.ones((495,1))


# In[136]:


# singer


# In[137]:


X_val_singer = np.stack(mel_spectrogram_val_singer,axis=0)
X_val_singer.shape


# In[138]:


X_val_singer_resh = X_val_singer.reshape(284,129,129,-1)


# In[139]:


X_val_singer_resh.shape


# In[140]:


y_val_singer = np.ones((284,1))*2


# In[141]:


# piano


# In[142]:


X_val_piano = np.stack(mel_spectrogram_val_piano,axis=0)
X_val_piano.shape


# In[143]:


X_val_piano_resh = X_val_piano.reshape(339,129,129,1)


# In[144]:


X_val_piano_resh.shape


# In[145]:


y_val_piano = np.ones((339,1))*3


# In[ ]:





# In[ ]:





# In[147]:


# X_val shape (number wav files, f domain, t domain, 1)
# number wav guitar + singer:
(328 + 495 + 284 + 339, 129, 129)


# In[148]:


X_val = np.concatenate([X_val_clarinet_resh, X_val_guitar_resh, X_val_singer_resh,X_val_piano_resh])


# In[149]:


X_val.shape


# In[150]:


y_val = np.concatenate([y_val_clarinet, y_val_guitar, y_val_singer,y_val_piano])


# In[151]:


y_val_clarinet[327][0]


# In[152]:


y_val_guitar[394][0]


# In[153]:


y_val_singer[283][0]


# In[154]:


y_val_piano[338,0]


# In[155]:


y_val.shape


# In[ ]:





# In[156]:


# One hot encode the labels:
# Keep a copy of y test as the actual number labels:
y_val_true = y_val.copy()


# In[157]:


y_val = to_categorical(y_val)


# In[158]:


y_val.shape


# In[159]:


y_val


# In[160]:


y_val[327:329,0]


# In[161]:


y_val[327:329,1]


# In[162]:


# the same needs to be done with the test 


# In[ ]:





# In[ ]:





# MODELLING

# In[163]:


# general functions and plots


# In[164]:


def plot_accuracy(history):    
    # Plot accuracy curve
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label= 'validation accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()


# In[165]:


def plot_loss(history):
# Plot loss
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()


# In[166]:


# Early stopping:
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', #the thing we are monitoring
    min_delta=0.0005, #the minimum change in the quantity that we want for the model to train for another epoch
    patience=3, #number of epochs with no improvement needed for the model to stop
    verbose=1, #0 is silent, 1 means a message is displayed when something happens
    mode='auto'
)


# In[ ]:


# Model


# In[323]:


#model 00: relu, relu


# In[330]:


# model 00
# Clear session and build above model:
K.clear_session()
model_00 = Sequential([
    Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1), padding = 'valid',
           activation = 'relu' ,input_shape = (129,129,1)),
    MaxPooling2D(pool_size = (2,2),strides=(2,2),padding = 'valid'),

    
    Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1), padding = 'valid', activation = 'relu'),
    MaxPooling2D(pool_size = (2,2),strides=(2,2),padding = 'valid'),
    
    Flatten(),
    #Fully connected layers:
    Dense(100,activation='relu'),
    #Dropout(0.2), - optional
    Dense(4,activation= 'softmax')   
])


# In[325]:


# Check the summary:
model_00.summary()


# In[326]:


# compile model
model_00.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[327]:


X_train.shape, y_train.shape


# In[328]:


X_val.shape, y_val.shape


# In[ ]:


# fit the model
history_00 = model_00.fit(X_train,y_train,batch_size=400, epochs=35,validation_data=(X_val,y_val))


# In[ ]:


model_00.save('../data/models/three_instruments/2conv_00.h5')


# In[ ]:


with open('../data/models/three_instruments/2conv_00_hist.h5', 'wb') as file_pi:
    pickle.dump(history_00.history, file_pi)


# In[ ]:


# history_2B_loaded = pickle.load(open('../data/models/two_instruments/3conv2d_bn_dout_hist.h5',"rb"))


# In[ ]:


# examine the model


# In[ ]:


history_00.history;


# In[ ]:


plot_accuracy(history_00)


# In[ ]:


plot_loss(history_00)


# In[ ]:


pd.DataFrame(history_00.history).plot()


# In[372]:


# model 0a: relu / tanh


# In[373]:


# model 0a
# Clear session and build above model:
K.clear_session()
model_0a = Sequential([
    Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1), padding = 'valid',
           activation = 'relu' ,input_shape = (129,129,1)),
    MaxPooling2D(pool_size = (2,2),strides=(2,2),padding = 'valid'),

    
    Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1), padding = 'valid', activation = 'tanh'),
    MaxPooling2D(pool_size = (2,2),strides=(2,2),padding = 'valid'),
    
    Flatten(),
    #Fully connected layers:
    Dense(100,activation='relu'),
    #Dropout(0.2), - optional
    Dense(4,activation= 'softmax')   
])


# In[374]:


# Check the summary:
model_0a.summary()


# In[375]:


# compile model
model_0a.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[376]:


# fit the model
history_0a = model_0a.fit(X_train,y_train,batch_size=722,epochs=40,validation_data=(X_val,y_val),)


# In[ ]:


model_0a.save('../data/models/four_instruments/2conv_0a.h5')


# In[178]:


model_0a_loaded = keras.models.load_model('../data/models/four_instruments/2conv_0a.h5')


# In[182]:


history_0a_loaded = model_0a_loaded.history


# In[190]:


dir(history_0a_loaded)


# In[185]:


plt.plot(model_0a_loaded.history)


# In[ ]:





# In[378]:


# examine the model


# In[ ]:


history_0a.history;


# In[380]:


plot_accuracy(history_0a)


# In[381]:


plot_loss(history_0a)


# In[382]:


pd.DataFrame(history_0a.history).plot()


# In[ ]:


#model 0b: tanh / tanh


# In[ ]:


# model 00b:
# Clear session and build above model:
K.clear_session()
model = Sequential([
    Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1), padding = 'valid',
           activation = 'tanh' ,input_shape = (129,129,1)),
    MaxPooling2D(pool_size = (2,2),strides=(2,2),padding = 'valid'),

    
    Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1), padding = 'valid', activation = 'tanh'),
    MaxPooling2D(pool_size = (2,2),strides=(2,2),padding = 'valid'),
    
    Flatten(),
    #Fully connected layers:
    Dense(100,activation='relu'),
    #Dropout(0.2), - optional
    Dense(2,activation= 'sigmoid')   
])


# In[ ]:


# Check the summary:
model.summary()


# In[ ]:


# compile model
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


# fit the model
history = model.fit(X_train,y_train,batch_size=400, epochs=35,validation_data=(X_val,y_val))


# In[ ]:


# examine the model


# In[ ]:


history.history;


# In[ ]:


plot_accuracy(history)


# In[ ]:


plot_loss(history)


# In[ ]:


pd.DataFrame(history.history).plot()


# In[167]:


# model 01:
# 2nd Conv2D with tanh
# bath norm after 1st Conv2D


# In[172]:


# model 01
# Clear session and build above model:
K.clear_session()

model_01 = Sequential([
    Conv2D( filters = 16, kernel_size = (3,3), strides = (1,1), padding = 'valid',
           activation = 'relu', input_shape = (129,129,1)),
    Dropout(0.3),
    BatchNormalization(),
    
    MaxPooling2D(pool_size = (2,2),strides=(2,2),padding = 'valid'),
    
    Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1), padding = 'valid', activation = 'tanh'),
    Dropout(0.3),
    MaxPooling2D(pool_size = (2,2),strides=(2,2),padding = 'valid'),
    
    Flatten(),
    #Fully connected layers:
    Dense(100,activation='relu'),
    #Dropout(0.2), - optional
    Dense(4,activation= 'softmax')   
])


# In[173]:


# Check the summary:
model_01.summary()


# In[174]:


# compile model
model_01.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[175]:


# fit the model
history_01 = model_01.fit(X_train,y_train,batch_size=750, epochs=40,validation_data=(X_val,y_val))


# In[ ]:


# examine the model


# In[ ]:


history.history


# In[ ]:


# Plot accuracy curve
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label= 'validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()


# In[ ]:


# Plot loss
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()


# In[ ]:


pd.DataFrame(history.history).plot()


# In[ ]:


# model 2
# 3 Conv2D


# In[ ]:


# model 02
# Clear session and build above model:
K.clear_session()

model_02 = Sequential([
    Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1), padding = 'valid',
           activation = 'relu' ,input_shape = (129,129,1)),
    
    MaxPooling2D(pool_size = (2,2),strides=(2,2),padding = 'valid'),
    
    Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1), padding = 'valid', activation = 'tanh'),
    
    MaxPooling2D(pool_size = (2,2),strides=(2,2),padding = 'valid'),
    
    Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1), padding = 'valid', activation = 'tanh'),
    
    MaxPooling2D(pool_size = (2,2),strides=(2,2),padding = 'valid'),
    
    Flatten(),
    #Fully connected layers:
    Dense(100,activation='relu'),
    #Dropout(0.2), - optional
    Dense(3,activation= 'softmax')   
])


# In[ ]:


# Check the summary:
model_02.summary()


# In[ ]:


# compile model
model_02.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


# fit the model
history_02 = model_02.fit(X_train,y_train,batch_size=400, epochs=40,validation_data=(X_val,y_val))


# In[ ]:


model_02.save('../data/models/three_instruments/3conv_02.h5')


# In[ ]:


# examine the model


# In[ ]:


history_02.history;


# In[ ]:


plot_accuracy(history_02)


# In[ ]:


plot_loss(history_02)


# In[ ]:


pd.DataFrame(history_02.history).plot()


# In[ ]:


new_model = keras.models.load_model('../data/models/two_instruments/3conv2d.h5')


# In[ ]:


new_model.summary()


# In[ ]:


print(new_model.history)


# FINAL MODEL FOR 2-Instrumention: 3 Conv2D with batch norm & dropout

# In[ ]:


# model 2B
# 3 Conv2D with batch norm & dropout


# In[ ]:


# model 2B
# Clear session and build above model:
K.clear_session()

model_2B = Sequential([
    Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1), padding = 'valid',
           activation = 'relu' ,input_shape = (129,129,1)),
    
    BatchNormalization(),
    
    MaxPooling2D(pool_size = (2,2),strides=(2,2),padding = 'valid'),
    
    Dropout(0.2),
    
    Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1), padding = 'valid', activation = 'tanh'),
    
    BatchNormalization(),
    
    MaxPooling2D(pool_size = (2,2),strides=(2,2),padding = 'valid'),
    
    Dropout(0.2),
    
    Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1), padding = 'valid', activation = 'tanh'),
    
    BatchNormalization(),
    
    MaxPooling2D(pool_size = (2,2),strides=(2,2),padding = 'valid'),
    
    Dropout(0.2),
    
    Flatten(),
    #Fully connected layers:
    Dense(100,activation='relu'),
    Dropout(0.2),
    Dense(3,activation= 'softmax')   
])


# In[ ]:


# Check the summary:
model_2B.summary()


# In[ ]:


# compile model
model_2B.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


# fit the model
history_2B = model_2B.fit(X_train,y_train,batch_size=700, epochs=40,validation_data=(X_val,y_val))


# In[ ]:


model_2B.save('../data/models/two_instruments/3conv2d_2B.h5')


# In[ ]:


model_2b_loaded = keras.models.load_model('../data/models/two_instruments/3conv2d_2B.h5')


# In[ ]:


with open('../data/models/two_instruments/3conv2d_2B_hist.h5', 'wb') as file_pi:
        pickle.dump(history_2B.history, file_pi)


# In[ ]:


history_2B_loaded = pickle.load(open('../data/models/two_instruments/3conv2d_2B_hist.h5',"rb"))


# In[ ]:


history_2B_loaded


# In[ ]:


history_2B


# In[ ]:


history_2B.history


# In[ ]:


# examine the model


# In[ ]:


plot_accuracy(history_2B)


# In[ ]:


plot_loss(history_2B)


# In[ ]:


# f, (ax1,ax2) = plt.subplots(1,2)
# ax1.plot.history_2B


# In[ ]:


pd.DataFrame(history_2B.history).plot()


# In[ ]:


# predictions:


# In[ ]:


y_val.shape


# In[ ]:


X_val[0].reshape(1,129,129,1).shape


# In[ ]:


y_pred = model_2B.predict(X_val[1].reshape(1,129,129,1))


# In[ ]:


y_pred


# In[ ]:


X_test_guitar, X_test_singer, X_clarinet


# In[ ]:





# In[ ]:


# model 03: more filters


# In[ ]:


# model 03
# Clear session and build above model:
K.clear_session()

model = Sequential([
    Conv2D( filters = 20, kernel_size = (3,3), strides = (1,1), padding = 'valid',
           activation = 'relu', input_shape = (129,129,1)),
    
    MaxPooling2D(pool_size = (2,2),strides=(2,2),padding = 'valid'),
    
    Conv2D(filters = 20, kernel_size = (3,3), strides = (1,1), padding = 'valid', activation = 'tanh'),
    
    MaxPooling2D(pool_size = (2,2),strides=(2,2),padding = 'valid'),
    
    Flatten(),
    #Fully connected layers:
    Dense(100,activation='relu'),
    #Dropout(0.2), - optional
    Dense(2,activation= 'sigmoid')   
])


# In[ ]:


# Check the summary:
model.summary()


# In[ ]:


# compile model
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


# fit the model
history = model.fit(X_train,y_train,batch_size=722, epochs=40,validation_data=(X_val,y_val))


# In[ ]:


# examine the model


# In[ ]:


history.history;


# In[ ]:


plot_accuracy(history)


# In[ ]:


plot_loss(history)


# In[ ]:


pd.DataFrame(history.history).plot()


# In[ ]:





# In[ ]:


# confusion matrix


# In[ ]:





# #### Pre trained CNN

# In[ ]:





# In[ ]:


# bonus: shuffling an array


# In[ ]:


abc = np.array([[1,2],[3,4],[5,6],[7,8],[0,9]])


# In[ ]:


abc


# In[ ]:


abc.shape


# In[ ]:


np.random.shuffle(abc)
abc


# In[ ]:


np.arange(10)


# In[ ]:




