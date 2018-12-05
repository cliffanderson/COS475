import tensorflow as tf
from tensorflow import keras

import numpy as numpy
import matplotlib.pyplot as plt

import sys, os, struct
from array import array
from sys import path

import pandas
from keras.layers import Input, Dense, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


print("TensorFlow version: ", tf.__version__)


#File names for training and test sets
train_images_fn = 'train-images.npy'
train_images_low_res_fn = 'train-images-low-res.npy'

test_images_fn = 'test-images.npy'
test_images_low_res_fn = 'test-images-low-res.npy'


#Load data into numpy arrays
low_res_image_array = numpy.load(train_images_low_res_fn)
image_array = numpy.load(train_images_fn)

test_low_res_image_array = numpy.load(test_images_low_res_fn)
test_image_array = numpy.load(test_images_fn)

print(low_res_image_array.shape)
print(image_array.shape)
print(test_low_res_image_array.shape)
print(test_image_array.shape)



# Normalize pixel values to fit range (0 - 1)
low_res_image_array = low_res_image_array / 255.0
image_array = image_array / 255.0

test_low_res_image_array = test_low_res_image_array / 255.0
test_image_array = test_image_array / 255.0


#Configure the neural network model
model = keras.Sequential(
    [
        keras.layers.Dense(14*14),
        keras.layers.Dense(19*19, activation='sigmoid'),
        keras.layers.Dense(24*24, activation='sigmoid'),
        keras.layers.Dense(28*28, activation='sigmoid')
    ]
)

#Compile and fit the model to the training data
model.compile(optimizer=keras.optimizers.Adam(), loss='mse', metrics=['mse'])
model.fit(low_res_image_array, image_array, epochs=1)

#Test our model against the test set
test_loss, test_acc = model.evaluate(test_low_res_image_array, test_image_array)

print('test loss: ', test_loss)
print('test accuracy: ', test_acc)

#Generate higher-resolution images in an array of predictions

predictions = model.predict(test_low_res_image_array);

print('Predictions: ', len(predictions), ' x ', len(predictions[0]))

# Export image data to a text file to feed into Java program

image_num = 3

an_image = predictions[image_num]
an_image = an_image * 255.0
f = open('new.txt', 'w')

for i in range(0, 28*28):
    # print(an_image[i])
    f.write(str(an_image[i]))
    f.write('\n')

f.close()


f = open('orig.txt', 'w')

orig_image = test_low_res_image_array[image_num]
orig_image = orig_image * 255.0

for i in range(0, 14*14):
    f.write(str(orig_image[i]))
    # print(orig_image[i])
    f.write('\n')

f.close()


f = open('orig_full_res.txt', 'w')

orig_full_res = test_image_array[image_num]
orig_full_res = orig_full_res * 255.0

for i in range(0, 28*28):
    f.write(str(orig_full_res[i]))
    f.write('\n')

f.close()