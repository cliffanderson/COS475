import os, struct
import numpy as np
from array import array


def load_training_images(file_name):
    training_images_fh = open(file_name, 'rb')
    magic_number, size, rows, columns = struct.unpack(">IIII", training_images_fh.read(16))
    _training_images = array("B", training_images_fh.read())
    training_images_fh.close()

    return _training_images


def load_testing_images(file_name):
    testing_images_fh = open(file_name, 'rb')
    magic_number, size, rows, columns = struct.unpack(">IIII", testing_images_fh.read(16))
    _testing_images = array("B", testing_images_fh.read())
    testing_images_fh.close()

    return _testing_images


def format_training_images(_training_images):
    image_array = np.zeros((60000, 28, 28))

    for image in range(0, 60000):
        for y in range(0, 28):
            for _x in range(0, 28):
                image_array[image][_x][y] = _training_images[image*784 + y * 28 + _x]

    # thestr = ''
    # for y in range(0, 28):
    #     for x in range(0, 28):
    #         thestr = thestr + str(image_array[0][x][y])
    #
    # print('rbg values: ', thestr)

    return image_array


def format_testing_images(_testing_images):
    image_array = np.zeros((10000, 28, 28))

    for image in range(0, 10000):
        for y in range(0, 28):
            for _x in range(0, 28):
                image_array[image][_x][y] = _testing_images[image*784 + y * 28 + _x]

    return image_array


def create_low_res_training_images(orig_training_images):
    low_res_image_array = np.zeros((60000, 14, 14))

    for image in range(0, 60000):
        for y in range(0, 14):
            for x in range(0, 14):
                # a = orig_training_images[image][x*2 + 1][y*2]
                # b = orig_training_images[image][x*2][y*2 + 1]
                # c = orig_training_images[image][x*2 + 1][y*2 + 1]
                # d = orig_training_images[image][x*2][y*2]
                #
                # print('a:', a, ' b:', b, ' c:', c, ' d:', d)
                # print(image, ' ', x, ' ', y)
                low_res_image_array[image][x][y] = (orig_training_images[image][x*2 + 1][y*2] +
                                                    orig_training_images[image][x*2][y*2 + 1] +
                                                    orig_training_images[image][x*2 + 1][y*2 + 1] +
                                                    orig_training_images[image][x*2][y*2]) / 4

    return low_res_image_array


def create_low_res_testing_images(orig_testing_images):
    low_res_image_array = np.zeros((10000, 14, 14))

    for image in range(0, 10000):
        for y in range(0, 14):
            for x in range(0, 14):
                # a = orig_training_images[image][x*2 + 1][y*2]
                # b = orig_training_images[image][x*2][y*2 + 1]
                # c = orig_training_images[image][x*2 + 1][y*2 + 1]
                # d = orig_training_images[image][x*2][y*2]
                #
                # print('a:', a, ' b:', b, ' c:', c, ' d:', d)
                # print(image, ' ', x, ' ', y)
                low_res_image_array[image][x][y] = (orig_testing_images[image][x*2 + 1][y*2] +
                                                    orig_testing_images[image][x*2][y*2 + 1] +
                                                    orig_testing_images[image][x*2 + 1][y*2 + 1] +
                                                    orig_testing_images[image][x*2][y*2]) / 4

    return low_res_image_array


training_images_file = 'train-images'
testing_images_file = 'test-images'

training_images_saved_array_file_name = 'train-images.npy'
training_images_low_res_saved_array_file_name = 'train-images-low-res.npy'

testing_images_saved_array_file_name = 'test-images.npy'
testing_images_low_res_saved_array_file_name = 'test-images-low-res.npy'

# Remove numpy files if they exist
if os.path.isfile(training_images_saved_array_file_name):
    os.remove(training_images_saved_array_file_name)

if os.path.isfile(training_images_low_res_saved_array_file_name):
    os.remove(training_images_low_res_saved_array_file_name)

if os.path.isfile(testing_images_saved_array_file_name):
    os.remove(testing_images_saved_array_file_name)

if os.path.isfile(testing_images_low_res_saved_array_file_name):
    os.remove(testing_images_low_res_saved_array_file_name)

# Load & process
training_images = load_training_images(training_images_file)
training_images_formatted = format_training_images(training_images)
training_images_low_res = create_low_res_training_images(training_images_formatted)

testing_images = load_testing_images(testing_images_file)
testing_images_formatted = format_testing_images(testing_images)
testing_images_low_res = create_low_res_testing_images(testing_images_formatted)


# from [60000, 28, 28] to [60000, 784]
new_training_images_formatted = np.zeros((60000, 784))
for image in range(0, 60000):
    for y in range(0, 28):
        for x in range(0, 28):
            new_training_images_formatted[image][y*28 + x] = training_images_formatted[image][x][y]

training_images_formatted = new_training_images_formatted


# from [60000, 14, 14] to [60000, 196]
new_training_images_low_res = np.zeros((60000, 196))
for image in range(0, 60000):
    for y in range(0, 14):
        for x in range(0, 14):
            new_training_images_low_res[image][y*14 + x] = training_images_low_res[image][x][y]

training_images_low_res = new_training_images_low_res


# from [60000, 28, 28] to [60000, 784]
new_testing_images_formatted = np.zeros((10000, 784))
for image in range(0, 10000):
    for y in range(0, 28):
        for x in range(0, 28):
            new_testing_images_formatted[image][y*28 + x] = testing_images_formatted[image][x][y]

testing_images_formatted = new_testing_images_formatted


# from [60000, 14, 14] to [60000, 196]
new_testing_images_low_res = np.zeros((10000, 196))
for image in range(0, 10000):
    for y in range(0, 14):
        for x in range(0, 14):
            new_testing_images_low_res[image][y*14 + x] = testing_images_low_res[image][x][y]

testing_images_low_res = new_testing_images_low_res


# Save
np.save(training_images_saved_array_file_name, training_images_formatted)
np.save(training_images_low_res_saved_array_file_name, training_images_low_res)
np.save(testing_images_saved_array_file_name, testing_images_formatted)
np.save(testing_images_low_res_saved_array_file_name, testing_images_low_res)