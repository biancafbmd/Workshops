
# coding: utf-8

# # Load the dataset

# In[1]:


#import the libraries

import keras

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# load the image data
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# # Exploratory Analysis

# In[4]:


fig = plt.figure(figsize=(5,5))

for i in range(1,21):
    fig.add_subplot(4, 5, i)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i-1])
    
plt.show()


# # Pre-process the data

# In[5]:


# pre-process the data
num_classes = 10
input_size = 28*28

# flatten the images
x_test_copy = x_test.copy()
x_train = x_train.reshape(x_train.shape[0], input_size)
x_test = x_test.reshape(x_test.shape[0], input_size)

# convert the labels to one-hot-encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[6]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_test_copy.shape)


# # Define the Neural Network Architecture

# In[7]:


# define your neural network architecture

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))


# In[8]:


model.summary()


# # Train the model

# In[9]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=512)


# # Evaluate the model

# In[10]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[11]:


y_pred = model.predict_classes(x_test[0:20])

fig = plt.figure(figsize=(5,5))

for i in range(1,21):
    fig.add_subplot(4, 5, i)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test_copy[i-1])
    
    plt.title("pred: %s" % (y_pred[i-1]),  fontsize=9, loc='left')
    
plt.show()

