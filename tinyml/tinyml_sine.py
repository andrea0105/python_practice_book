import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt

# Different values for each computing
samples = 1000
seed = 1330

np.random.seed(seed)
tf.random.set_seed(seed)

# Create uniformly distributed random numberset between 0 - 2pi
x_values = np.random.uniform(low=0, high=2*math.pi, size=samples)

# Shuffle it not to follow orders 
np.random.shuffle(x_values)

# Calculation
y_values = np.sin(x_values)

# Add arbitrary small number, regarding as a noise
y_values += 0.1 * np.random.randn(*y_values.shape)

# Data need to be trisected.
# 60% for training, 20% for testing, 20% for validating
# Calculating the index for each sector.
TRAIN_SPLIT = int(0.6 * samples)
TEST_SPLIT = int(0.2 * samples + TRAIN_SPLIT)

# By using np.split, cut the data into trisector.
# Second parameter for np.split is the index array
# Use two index, three lumps of data
x_train, x_validate, x_test = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_validate, y_test = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

# When amalgamating data, checking it with original size
assert (x_train.size + x_validate.size + x_test.size) == samples

# Plotting, all data in different colors
plt.plot(x_train, y_train, 'b.', label="Train")
plt.plot(x_validate, y_validate, 'y.', label="Validate")
plt.plot(x_test, y_test, 'r.', label="Test")
plt.legend()
plt.savefig("Data_plot.png")
plt.show()
