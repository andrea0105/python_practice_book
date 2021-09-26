import tensorflow as tf
from tensorflow.keras import layers
import math
import numpy as np
import matplotlib.pyplot as plt

#--------------------------Data Generation---------------------------------#

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

#-------------------------Basic Model Define-------------------------------#

# To build simple model structure, use keras
model_1 = tf.keras.Sequential()

# First layer has sixteen neurons and convey scalar inputs to the nex layer
# Neuron determine whether to convey according to the 'relu' function
# Single Input(x_values) and 16 neurons -> Dense layer
# Dense layer -> all inputs go into one node
# Level of neuron's activation is based on a weight,bias and activation function
# Neurons's activation expressed as a number

model_1.add(layers.Dense(16, activation = 'relu', input_shape = (1,)))

# Final layer has one neuron for one output in last
model_1.add(layers.Dense(1))

# By using standard optimizer and loss, compiling a regression model
model_1.compile(optimizer = 'rmsprop', loss='mse', metrics=['mae'])

# Give a summarized model design
model_1.summary()

#--------------------------Improved Model----------------------------------#

model_2 = tf.keras.Sequential()

# 16 Inputs -> 16 neurons
# Activation function = relu
model_2.add(layers.Dense(16, activation='relu', input_shape=(1,)))

# New second layer helps networks to learn much complicated expression 
model_2.add(layers.Dense(16, activation='relu'))

# Final layer is one neuron to make one output
model_2.add(layers.Dense(1))

# Compile a regression model with standard optimizer and loss function
model_2.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# Model Summary
model_2.summary()

#---------------------------Train a Model----------------------------------#

history_2 = model_2.fit(x_train, y_train, epochs = 600, batch_size = 16, \
        validation_data=(x_validate, y_validate))

#-----------------------------Plotting-------------------------------------#

plt.clf()

# Another way of calculating, mean absolute error graph

mae = history_2.history['mae']
val_mae = history_2.history['val_mae']

epochs = range(1, len(mae) + 1)
skip = 100
loss = model_2.evaluate(x_test, y_test)
predictions = model_2.predict(x_test)
plt.title('Comparison of predictions and actual values')
plt.plot(x_test, y_test, 'b.', label='Actual')
plt.plot(x_test, predictions, 'r.', label='Predicted')
plt.legend()
plt.savefig('Model2_comparison.png')
plt.show()
