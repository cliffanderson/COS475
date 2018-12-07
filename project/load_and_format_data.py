import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.datasets import cifar10

### File names (to store numpy arrays to save time) ###

# Numpy files where MNIST dataset is stored ()
train_fn = 'data/train.npy'
test_fn = 'data/test.npy'
low_res_train_fn = 'data/low-res-train.npy'
low_res_test_fn = 'data/low-res-test.npy'

# Unwrapped (4D) arrays for CIFAR-10 (matplotlib input)
c_train_fn = 'data/c-train.npy'
c_test_fn = 'data/c-test.npy'
c_low_res_train_fn = 'data/c-low-res-train.npy'
c_low_res_test_fn = 'data/c-low-res-test.npy'


# Load the MNIST dataset and returns it as a (60000, 784) and a (10000, 784) numpy array
def load_images():

    mnist = tf.keras.datasets.mnist
    (image_array, y_train),(test_image_array, y_test) = mnist.load_data()

    train_data = np.array(image_array)
    test_data = np.array(test_image_array)

    (size, l, w) = image_array.shape

    if not os.path.isfile(train_fn):
        train_data = np.reshape(train_data, (size, (l * w),))
        np.save(train_fn, train_data)
    else:
        train_data = np.load(train_fn)

    (size, l, w) = test_image_array.shape

    if not os.path.isfile(test_fn):
        test_data = np.reshape(test_data, (size, (l * w),))
        np.save(test_fn, test_data)
    else:
        test_data = np.load(test_fn)

    return (train_data, test_data)


# Load the low resolution version of the MNIST dataset
def load_low_res_images():

    mnist = tf.keras.datasets.mnist
    (image_array, y_train),(test_image_array, y_test) = mnist.load_data()

    return (load_or_generate_low_res_images(low_res_train_fn, image_array), \
        load_or_generate_low_res_images(low_res_test_fn, test_image_array))


# Load low resolution version of MNIST dataset, or generate it if the file doesn't exist
def load_or_generate_low_res_images(fn, data):

    low_res_data = None

    if not os.path.isfile(fn):
        low_res_data = create_low_res_images(data)
        (size, l, w) = low_res_data.shape
        low_res_data = np.reshape(low_res_data, (size, (l * w),))
        np.save(fn, low_res_data)
    else:
        low_res_data = np.load(fn)

    return low_res_data


# Use original images to generate low resolution version
def create_low_res_images(image_data):

    size, orig_l, orig_w = image_data.shape;
    l, w = int(orig_l / 2), int(orig_w / 2)

    low_res_image_array = np.zeros((size, l, w))

    #Generate compressed version of original dataset
    for x in range(size):
        y = 0
        z = 0
        while z < w:
            while y < l:
                low_res_image_array[x][y][z] = (int(image_data[x][(y * 2) + 1][(z * 2) + 1]) \
                + int(image_data[x][(y * 2) + 1][z * 2]) \
                + int(image_data[x][y * 2][(z * 2) + 1]) \
                + int(image_data[x][y * 2][z * 2])) / 4.0
                y += 1
            y = 0
            z += 1

    return low_res_image_array


def load_cifar10():

    (cifar_x_train, cifar_y_train), (cifar_x_test, cifar_y_test) = cifar10.load_data()
    train_data, test_data = None, None

    if os.path.isfile(c_train_fn) and os.path.isfile(c_test_fn):
        train_data = np.load(c_train_fn)
        test_data = np.load(c_test_fn)
    else:
        (train_data, test_data) = (cifar_x_train, cifar_x_test)

        (size, l, w, ch) = train_data.shape
        train_data = np.reshape(train_data, (size, l * w * ch,))
        np.save(c_train_fn, train_data)

        (size, l, w, ch) = test_data.shape
        test_data = np.reshape(test_data, (size, l * w * ch,))
        np.save(c_test_fn, test_data)

    low_res_train_data, low_res_test_data = None, None

    if os.path.isfile(c_low_res_train_fn):
        low_res_train_data = np.load(c_low_res_train_fn)
    else:
        
        x_train = create_low_res_cifar10(cifar_x_train)
        (size, l, w, ch) = x_train.shape
        low_res_train_data = np.reshape(x_train, (size, l * w * ch,))
        np.save(c_low_res_train_fn, low_res_train_data)

    if os.path.isfile(c_low_res_test_fn):
        low_res_test_data = np.load(c_low_res_test_fn)
    else:
        x_test = create_low_res_cifar10(cifar_x_test)
        (size, l, w, ch) = x_test.shape
        low_res_test_data = np.reshape(x_test, (size, l * w * ch,))
        np.save(c_low_res_test_fn, low_res_test_data)

    return (train_data, test_data), (low_res_train_data, low_res_test_data)


    return low_res_image_array

    # Use original images to generate low resolution version
def create_low_res_cifar10(image_data):

    size, orig_l, orig_w, _ch = image_data.shape;
    _l, _w = int(orig_l / 2), int(orig_w / 2)

    low_res_image_array = np.zeros((size, _l, _w, _ch))

    #Generate compressed version of original dataset
    for image in range(size):
        for ch in range(_ch):
            l = 0
            w = 0
            while w < _w:
                while l < _l:
                    low_res_image_array[image][l][w][ch] = (int(image_data[image][(l * 2) + 1][(w * 2) + 1][ch]) \
                    + int(image_data[image][(l * 2) + 1][w * 2][ch]) \
                    + int(image_data[image][l * 2][(w * 2) + 1][ch]) \
                    + int(image_data[image][l * 2][w * 2][ch])) / 4.0
                    l += 1
                l = 0
                w += 1

    return low_res_image_array