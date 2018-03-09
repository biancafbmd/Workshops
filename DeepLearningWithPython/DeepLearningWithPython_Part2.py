
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


# # Pre-process the data

# In[4]:


fig = plt.figure(figsize=(5,5))

for i in range(1,21):
    fig.add_subplot(4, 5, i)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i-1])
    
plt.show()


# In[5]:


# pre-process the data
num_classes = 10

# convert the labels to one-hot-encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# reshape to (rows, columns, channels)
x_test_copy = x_test.copy()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# # Define the Neural Network Architecture

# In[6]:


# define your neural network architecture

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten

input_shape = (28, 28, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(num_classes, activation='softmax'))


# In[7]:


model.summary()


# # Train the model

# In[8]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=128)


# # Evaluate the model

# In[9]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[10]:


y_pred = model.predict_classes(x_test[0:20])

fig = plt.figure(figsize=(5,5))

for i in range(1,21):
    fig.add_subplot(4, 5, i)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test_copy[i-1])
    
    plt.title("pred: %s" % (y_pred[i-1]),  fontsize=9, loc='left')
    
plt.show()

