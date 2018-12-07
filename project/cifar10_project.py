import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import load_and_format_data as lf


print("TensorFlow version: ", tf.__version__)

# Load CIFAR-10 dataset (ignore labels)
print("Loading CIFAR-10...")
(train_data, test_data), (low_res_train_data, low_res_test_data) = lf.load_cifar10()


# Normalize pixel values...
print("Normalizing pixel values...")
train_data, test_data = train_data / 255.0, test_data / 255.0
low_res_train_data, low_res_test_data = low_res_train_data / 255.0, low_res_test_data / 255.0


print("Configuring and training model...")

# Configure the neural network model
model = keras.Sequential(
    [
        keras.layers.Dense(16 * 16 * 3),
        keras.layers.Dense(24 * 24 * 3, activation='sigmoid'),
        keras.layers.Dense(28 * 28 * 3, activation='sigmoid'),
        keras.layers.Dense(32 * 32 * 3, activation='sigmoid')
    ]
)


# Compile and fit the model to the training data
model.compile(optimizer=keras.optimizers.Adam(), loss='mse', metrics=['mse'])
model.fit(low_res_train_data, train_data, epochs=3)


# Test our model against the test set
test_loss, test_acc = model.evaluate(low_res_test_data, test_data)
print('test loss: ', test_loss)
print('test accuracy: ', test_acc)


# Generate high-resolution predictions
predictions = model.predict(low_res_test_data);
print('Predictions: ', len(predictions), ' x ', len(predictions[0]))


# Unwrap images to displayable format
print("Unwrapping images...")
d_test_data= np.reshape(test_data, (10000, 32, 32, 3,))
d_low_res_test_data = np.reshape(low_res_test_data, (10000, 16, 16, 3,))
d_predictions = np.reshape(predictions, (10000, 32, 32, 3,))
#d_predictions = np.round_(d_predictions * 255, decimals=0).astype(int)


# Creates a plot to display our images
fig = plt.figure(figsize=(15, 15))
rows = 5
columns = 3

for i in range(0, rows):
    ax1 = fig.add_subplot(rows, columns, i * 3 + 1)
    plt.imshow(d_test_data[i])

    ax2 = fig.add_subplot(rows, columns, i * 3 + 2)
    plt.imshow(d_low_res_test_data[i])

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
matplotlib.image.imsave('images/orig.png', d_test_data[1])
matplotlib.image.imsave('images/comp.png', d_low_res_test_data[1])