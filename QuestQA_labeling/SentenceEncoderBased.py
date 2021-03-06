import numpy as np 
import pandas as pd 
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

import pickle    
import os

datadir = '../input/google-quest-challenge'

train = os.path.join(datadir,'train.csv')
test = os.path.join(datadir,'test.csv')
sample = os.path.join(datadir,'sample_submission.csv')

df_train = pd.read_csv(train)
df_test = pd.read_csv(test)
df_sample = pd.read_csv(sample)

target_columns = [col for col in df_train.columns if col in df_sample.columns] 

target_columns.remove('qa_id')  # targets
input_columns = ['question_title','question_body','answer'] # features

# google sentence model
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.load(module_url)


X1 = df_train[input_columns[0]].values.tolist()
X2 = df_train[input_columns[1]].values.tolist()
X3 = df_train[input_columns[2]].values.tolist()

X1 = [x.replace('?','.').replace('!','.') for x in X1]
X2 = [x.replace('?','.').replace('!','.') for x in X2]
X3 = [x.replace('?','.').replace('!','.') for x in X3]

def text2vec(t1,t2,t3):
  v1 = [embed([x]) for x in t1]
  v1 = [x.numpy()[0] for x in v1]
  v1 = np.array(v1)

  v2 = [embed([x]) for x in t2]
  v2 = [x.numpy()[0] for x in v2]
  v2 = np.array(v2)

  v3 = [embed([x]) for x in t3]
  v3 = [x.numpy()[0] for x in v3]
  v3 = np.array(v3)
  
  return v1,v2,v3

vec = text2vec(X1,X2,X3)


X = [vec[0],vec[1],vec[2]] # inputs
y = df_train[target_columns].values.tolist() # targets

# X,y  data type : numpy array
y = np.array(y)

# activation func
def swish(x):
  return K.sigmoid(x) * x


# ------------create network model---------------#

from tensorflow.keras import *
import tensorflow.keras.backend as K

embed_size = 512 # sentence encoder output is 512 dim

input1 = Input(shape=(embed_size,),name='X1')
input2 = Input(shape=(embed_size,),name='X2')
input3 = Input(shape=(embed_size,),name='X3')

# second layer
x = K.concatenate([input1,input2,input3]) 
x = layers.Dense(256,activation=swish)(x)  
x = layers.Dropout(0.2)(x)                 # Prevent overfitting
x = layers.BatchNormalization()(x)         

# third layer
x = layers.Dense(64,activation='relu')(x)   
x = layers.Dropout(0.2)(x)
x = layers.BatchNormalization()(x)

# output layer
output = layers.Dense(len(target_columns),activation='sigmoid',name='output')(x)

MODEL = Model(inputs=[input1,input2,input3],outputs=[output],name='ka1')  # multiple inputs = [..., ..., ... ]
MODEL.summary()

# ---------Train the network------------

reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=1e-7, verbose=1)
                              
optimizer = optimizers.Adadelta()
MODEL.compile(optimizer=optimizer, loss='binary_crossentropy')
MODEL.fit(X, y, epochs=30, validation_split=.1,batch_size=32,callbacks=[reduce_lr]) 


# -------predict test data----------------------

X_1 = df_test[input_columns[0]].values.tolist()
X_2 = df_test[input_columns[1]].values.tolist()
X_3 = df_test[input_columns[2]].values.tolist()

X_1 = [x.replace('?','.').replace('!','.') for x in X_1]
X_2 = [x.replace('?','.').replace('!','.') for x in X_2]
X_3 = [x.replace('?','.').replace('!','.') for x in X_3]

pred_vec = text2vec(X_1,X_2,X_3)
pred_X = [pred_vec[0], pred_vec[1], pred_vec[2]]  
pred_y = MODEL.predict(pred_X)

# file 
df_sample[target_columns] = pred_y   
df_sample.head()
df_sample.to_csv("submission.csv", index = False) 
