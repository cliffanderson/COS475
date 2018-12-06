import os
import tensorflow as tf
import numpy as np

# File names (to store numpy arrays to save time)
#Flattened (2D) arrays
train_fn = 'data/train-images.npy'
test_fn = 'data/test-images.npy'
low_res_train_fn = 'data/train-images-low-res.npy'
low_res_test_fn = 'data/test-images-low-res.npy'

#Unwrapped (3D) arrays
uw_test_fn = 'data/uw-test-images.npy'
uw_low_res_test_fn = 'data/uw-test-images-low-res.npy'
uw_predictions_fn = 'data/uw-predictions'


# Load the MNIST dataset
def load_images():

    mnist = tf.keras.datasets.mnist
    (image_array, y_train),(test_image_array, y_test) = mnist.load_data()

    train_data = None
    test_data = None

    if not os.path.isfile(train_fn):
        train_data = flatten_array(image_array)
        np.save(train_fn, train_data)
    else:
        train_data = np.load(train_fn)

    if not os.path.isfile(test_fn):
        test_data = flatten_array(test_image_array)
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
        low_res_data = flatten_array(create_low_res_images(data))
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


# Convert 3D array (img_num * l * w) to 2D (img_num * (l * w))   
def flatten_array(images):
    size = images.shape[0]
    l = images.shape[1]
    w = images.shape[2]

    formatted_array = np.zeros((size, l * w))
    for image in range(0, size):
        for y in range(0, l):
            for x in range(0, w):
                formatted_array[image][y * l + x] = images[image][x][y]

    return formatted_array


# Convert 2D array (img_num * (l * w)) to 3D (img_num * l * w) 
def unwrap_array(images, l, w, arr_type):
    
    fn = None

    # o (Original), c (Compressed), p (Prediction)
    if arr_type == 'o':
        fn = uw_test_fn
    elif arr_type == 'c':
        fn = uw_low_res_test_fn
    elif arr_type == 'p':
        fn = uw_predictions_fn
    else:
        fn = ""

    formatted_array = None
    size = images.shape[0]

    # If file does not exist, unwrap array manually
    if not os.path.isfile(fn) or arr_type == 'p':
        formatted_array = np.zeros((size, l, w))
        for image in range(0, size):
            for y in range(0, l):
                for x in range(0, w):
                    formatted_array[image][x][y] = images[image][y * l + x]

        np.save(fn, formatted_array)

    # If file exists, load numpy array from file
    else:
        formatted_array = np.load(fn)

    return formatted_array