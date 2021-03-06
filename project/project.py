import tensorflow as tf
from tensorflow import keras
import load_and_format_data as lf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


print("TensorFlow version: ", tf.__version__)

print("Loading MNIST...")

# Load MNIST dataset and compressed version
(image_array, test_image_array) = lf.load_images()
(low_res_image_array, test_low_res_image_array) = lf.load_low_res_images()

print("Normalizing pixel values...")

# Normalize pixel values
image_array, test_image_array = image_array / 255.0, test_image_array / 255.0
low_res_image_array, test_low_res_image_array = low_res_image_array / 255.0, test_low_res_image_array / 255.0

print("Configuring and training model...")

# Configure the neural network model
model = keras.Sequential(
    [
        keras.layers.Dense(14*14),
        keras.layers.Dense(19*19, activation='sigmoid'),
        keras.layers.Dense(24*24, activation='sigmoid'),
        keras.layers.Dense(28*28, activation='sigmoid')
    ]
)


# Compile and fit the model to the training data
model.compile(optimizer=keras.optimizers.Adam(), loss='mse', metrics=['mse'])
model.fit(low_res_image_array, image_array, epochs=3)


# Test our model against the test set
test_loss, test_acc = model.evaluate(test_low_res_image_array, test_image_array)
print('test loss: ', test_loss)
print('test accuracy: ', test_acc)


# Generate high-resolution predictions
predictions = model.predict(test_low_res_image_array);
print('Predictions: ', len(predictions), ' x ', len(predictions[0]))


# Unwrap images to displayable format
print("Unwrapping images...")

d_predictions = np.reshape(predictions, (predictions.shape[0], 28, 28,))
d_test_image_array = np.reshape(test_image_array, (test_image_array.shape[0], 28, 28,))
d_test_low_res_image_array = np.reshape(test_low_res_image_array, (test_low_res_image_array.shape[0], 14, 14))


# Creates a plot to display our images
fig = plt.figure(figsize=(15, 15))
rows = 5
columns = 3

for i in range(0, rows):
    ax1 = fig.add_subplot(rows, columns, i * 3 + 1)
    plt.imshow(d_test_image_array[i])

    ax2 = fig.add_subplot(rows, columns, i * 3 + 2)
    plt.imshow(d_test_low_res_image_array[i])

    ax3 = fig.add_subplot(rows, columns, i * 3 + 3)
    plt.imshow(d_predictions[i])

# Label columns
axes = fig.get_axes()
axes[0].set_title("Original")
axes[1].set_title("Compressed")
axes[2].set_title("Prediction")

plt.show()

# Export image data as .png
matplotlib.image.imsave('images/pred.png', d_predictions[1])
matplotlib.image.imsave('images/orig.png', d_test_image_array[1])
matplotlib.image.imsave('images/comp.png', d_test_low_res_image_array[1])