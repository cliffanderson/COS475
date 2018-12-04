import tensorflow as tf
from tensorflow import keras

import numpy as numpy
import matplotlib.pyplot as plt

import sys, os, struct
from array import array
from sys import path

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("TensorFlow version: ", tf.__version__)


# # Read in the dataset ### Credit: https://github.com/myleott/mnist_png/blob/master/convert_mnist_to_png.py
# training_labels_file = open('train-labels', 'rb')
# magic_number, size = struct.unpack(">II", training_labels_file.read(8))
# training_labels = array("b", training_labels_file.read())
# training_labels_file.close()
#
# # print('Labels')
# # print('\tMagic number: ', magic_number)
# # print('\tSize: ', size)
#
# training_images_file = open('train-images', 'rb')
# magic_number, size, rows, columns = struct.unpack(">IIII", training_images_file.read(16))
# training_images = array("B", training_images_file.read())
# training_images_file.close()
#
#
#
# test_images_file = open('test-images', 'rb')
# magic_number, size, rows, columns = struct.unpack(">IIII", test_images_file.read(16))
# test_images = array("B", test_images_file.read())
# test_images_file.close();
#
# print(len(test_images))
#
# # print('Images')
# # print('\tMagic number: ', magic_number)
# # print('\tSize: ', size)
# #
# # for i in range(1, 500):
# #     print(training_images[i])
#
#
# # image_array = [60000][28][28]
# image_array = numpy.zeros((60000, 28, 28))
#
# for image in range(0, 60000):
#     for y in range(0, 28):
#         for x in range(0, 28):
#             image_array[image][x][y] = training_images[image*784 + y * 28 + x]
#
# # print(len(image_array[0]))
#
# # Now produce lower-resolution versions of the image data above
# low_res_image_array = numpy.zeros((60000, 14, 14))
#
# for image in range(0, 60000):
#     for y in range(0, 14):
#         for x in range(0, 14):
#             low_res_image_array[image][x][y] = (image_array[image][x*2 + 1][y*2] + image_array[image][x*2][y*2 + 1] +
#                                                 image_array[image][x * 2 + 1][y * 2 + 1] + image_array[image][x*2][y*2]) / 4
#
#
# label_images = numpy.zeros((60000, 28*28))
# for image in range(0, 60000):
#     for i in range(0, 28*28):
#         label_images[image][i] = training_images[image*28*28 + i]
#
# #
#
# test_image_array = numpy.zeros((10000, 28, 28))
#
# for image in range(0, 10000):
#     for y in range(0, 28):
#         for x in range(0, 28):
#             test_image_array[image][x][y] = test_images[image*784 + y * 28 + x]
#
# # Now produce lower-resolution versions of the image data above
# test_low_res_image_array = numpy.zeros((10000, 14, 14))
#
# for image in range(0, 60000):
#     for y in range(0, 14):
#         for x in range(0, 14):
#             low_res_image_array[image][x][y] = (image_array[image][x*2 + 1][y*2] + image_array[image][x*2][y*2 + 1] +
#                                                 image_array[image][x * 2 + 1][y * 2 + 1] + image_array[image][x*2][y*2]) / 4





# Preprocess the data
# image_array = image_array / 255.0
# low_res_image_array = low_res_image_array / 255.0

# Setup neural network
# model = keras.Sequential(
#     [
#         keras.layers.Flatten(input_shape=(14, 14)),
#         # keras.layers.Input(shape=(196,)),
#         keras.layers.Dense(128, activation=tf.nn.relu),
#         keras.layers.Dense(28*28, activation=tf.nn.softmax)
#     ]
# )

training_images_saved_array_file_name = 'train-images.npy'
training_images_low_res_saved_array_file_name = 'train-images-low-res.npy'

testing_images_saved_array_file_name = 'test-images.npy'
testing_images_low_res_saved_array_file_name = 'test-images-low-res.npy'


low_res_image_array = numpy.load(training_images_low_res_saved_array_file_name)
image_array = numpy.load(training_images_saved_array_file_name)

testing_low_res_image_array = numpy.load(testing_images_low_res_saved_array_file_name)
testing_image_array = numpy.load(testing_images_saved_array_file_name)


# Normalize
low_res_image_array = low_res_image_array / 255.0
image_array = image_array / 255.0

testing_low_res_image_array = testing_low_res_image_array / 255.0
testing_image_array = testing_image_array / 255.0

from keras.layers import Input, Dense, Flatten
from keras.models import Model

activation_function = 'sigmoid'

model = keras.Sequential(
    [
        # keras.layers.Dense(14*14),
        # keras.layers.Dense(24*24, activation=activation_function),
        # keras.layers.Dense(28*28, activation=activation_function)

        keras.layers.Dense(14*14),
        keras.layers.Dense(19*19, activation=activation_function),
        keras.layers.Dense(24*24, activation=activation_function),
        keras.layers.Dense(28*28, activation=activation_function)
    ]
)

model.compile(optimizer=keras.optimizers.Adam(), loss='mse', metrics=['mse'])

model.fit(low_res_image_array, image_array, epochs=3)

test_loss, test_acc = model.evaluate(testing_low_res_image_array, testing_image_array)

print('Testing loss: ', test_loss)
print('Testing accuracy: ', test_acc)


# predictions = model.predict(low_res_image_array)
predictions = model.predict(testing_low_res_image_array)

print('Predictions: ', len(predictions), ' x ', len(predictions[0]))


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

orig_image = testing_low_res_image_array[image_num]
orig_image = orig_image * 255.0

for i in range(0, 14*14):
    f.write(str(orig_image[i]))
    # print(orig_image[i])
    f.write('\n')

f.close()


f = open('orig_full_res.txt', 'w')

orig_full_res = testing_image_array[image_num]
orig_full_res = orig_full_res * 255.0

for i in range(0, 28*28):
    f.write(str(orig_full_res[i]))
    f.write('\n')

f.close()



# model.compile(optimizer=tf.train.AdamOptimizer(),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(numpy.array(low_res_image_array), numpy.array(image_array), epochs=5)
# model.fit(low_res_image_array, image_array, epochs=5)


# test_loss, test_acc = model.evaluate(test_images, test_labels)

# print(test_loss)

# print('Test accuracy: ', test_acc)




# model = keras.Sequential(
#     [
#         keras.layers.Flatten(input_shape=(14, 14)),
#         # keras.layers.Dense(input_dim=14*14, kernel_initializer='normal', activation='relu'),
#         keras.layers.Dense(490, kernel_initializer='normal', activation='relu'),
#         keras.layers.Dense(28*28, kernel_initializer='normal')
#     ]
# )
#
# estimator = KerasRegressor(build_fn=model, epochs=100, batch_size=5, verbose=0)
#
# seed = 7
# numpy.random.seed(seed)
#
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(estimator, numpy.array(low_res_image_array), numpy.array(label_images), cv=kfold)
#
# # model.compile(loss='mean_squared_error', optimizer='adam')
#
# # model.fit(numpy.array(low_res_image_array), numpy.array(label_images), epochs=5)
# #
#
#
# predictions = model.predict(numpy.array(low_res_image_array))
#
# print(predictions[0])
#
# an_image = predictions[0]
# an_image = an_image * 255 # back to rgb-range
#
# thelist = an_image.tolist()
#
# for x in thelist:
#     print(x)
#
#
# import png
# f = open('digit.png', 'wb')
# w = png.Writer(28, 28, greyscale=True)
# w.write(f, an_image.tolist())
# f.close()