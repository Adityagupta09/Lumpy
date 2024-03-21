#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image

import pickle
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# In[3]:


data_dir = 'C:\\Users\\adity\\Downloads\\DATASET\\Cattles'


# In[4]:


datagenerator = {
    "train": ImageDataGenerator(horizontal_flip=True,
                                vertical_flip=True,
                                rescale=1. / 255,
                                validation_split=0.1,
                                shear_range=0.1,
                                zoom_range=0.1,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                rotation_range=30,
                               ).flow_from_directory(directory=data_dir,
                                                     target_size=(256, 256),
                                                     subset='training',
                                                    ),

    "valid": ImageDataGenerator(rescale=1 / 255,
                                validation_split=0.1,
                               ).flow_from_directory(directory=data_dir,
                                                     target_size=(256, 256),
                                                     subset='validation',
                                                    ),
}


# In[5]:


# Initializing InceptionV3 (pretrained) model with input image shape as (300, 300, 3)
base_model = VGG19(weights=None, include_top=False, input_shape=(256, 256, 3))


# In[18]:


import urllib.request

# Download the weights file

weights_file_path = 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Load the weights
base_model.load_weights(weights_file_path)
base_model.trainable = False


# In[19]:


model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.15),
    Dense(1024, activation='relu'),
    Dense(3, activation='softmax') # 10 Output Neurons for 10 Classes
])


# In[20]:


# Using the Adam Optimizer to set the learning rate of our final model
opt = optimizers.Adam(learning_rate=0.0001)

# Compiling and setting the parameters we want our model to use
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])


# In[21]:


model.summary()


# In[31]:


batch_size = 64
epochs = 10

# Seperating Training and Testing Data
train_generator = datagenerator["train"]
valid_generator = datagenerator["valid"]


# In[32]:


steps_per_epoch = train_generator.n // batch_size
validation_steps = valid_generator.n // batch_size

print("steps_per_epoch :", steps_per_epoch)
print("validation_steps :", validation_steps)


# In[33]:


from keras.callbacks import ModelCheckpoint

# File path to store the trained models
filepath = "./model_{epoch:02d}-{val_accuracy:.2f}.keras"

# Using the ModelCheckpoint function to train and store all the best models
checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Combining all callbacks into a list
callbacks_list = [checkpoint1]

# Training the Model
# Training the Model
history = model.fit(x=train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                    validation_data=valid_generator, validation_steps=validation_steps,
                    callbacks=callbacks_list)




# In[34]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# ________________ Graph 1 -------------------------

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

# ________________ Graph 2 -------------------------

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()


# In[35]:


test_loss, test_acc = model.evaluate(valid_generator)
print('test accuracy : ', test_acc)

