import tensorflow as tf
from tensorflow.keras import layers
import math
import numpy as np
import matplotlib.pyplot as plt
import os

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
y_values += 0.2 * np.random.randn(*y_values.shape)

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

history_2 = model_2.fit(x_train, y_train, epochs = 600, batch_size = 24, \
        validation_data=(x_validate, y_validate))

#------------------Conversion to Tensorflow Lite---------------------------#

# Without a quantumization, convert a model to a form of tf.lite
converter = tf.lite.TFLiteConverter.from_keras_model(model_2)
tflite_model = converter.convert()

#store a model to a disk
open("sine_model.tflite", "wb").write(tflite_model)

# Quantumize it and convert it to a form tf.lite
converter = tf.lite.TFLiteConverter.from_keras_model(model_2)

# Including a quantumization, performing a basic optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Define a generation function 
# which gives x_values of a validation date as a representative data
# Use this for a converter
def representative_dataset_generator():
        for value in x_test:
                # Each scalar value need to be in a two-dimension array stacked as a list
                yield[np.array(value, dtype = np.float32, ndmin = 2)]
converter.representative_dataset = representative_dataset_generator

# Model converting
tflite_model = converter.convert()

# Store a model in a disk
open("sine_model_quantized.tflite", "wb").write(tflite_model)

#-----------------------------Plotting-------------------------------------#

predictions = model_2.predict(x_test)

# Instantiation of an each model
sine_model = tf.lite.Interpreter('sine_model.tflite')
sine_model_quantized = tf.lite.Interpreter('sine_model_quantized.tflite')

# memory allocation to a each model
sine_model.allocate_tensors()
sine_model_quantized.allocate_tensors()

# Fetch a input tensor index and a output tensor index
sine_model_input_index = sine_model.get_input_details()[0]['index'] 
sine_model_output_index = sine_model.get_output_details()[0]['index']
sine_model_quantized_input_index = sine_model_quantized.get_input_details()[0]['index']
sine_model_quantized_output_index = sine_model_quantized.get_output_details()[0]['index']

# Create arrays to store outputs
sine_model_predictions = []
sine_model_quantized_predictions = []

# Perform an each interpreter for an each value and store outputs to arrays
for x_value in x_test:
        # Create a two-dimension tensor which envelopes a current x_value
        x_value_tensor = tf.convert_to_tensor([[x_value]], dtype = np.float32)

        # Write a value to an input tensor
        sine_model.set_tensor(sine_model_input_index, x_value_tensor)

        # Execute a prediction
        sine_model.invoke()
        
        # Read predictions
        sine_model_predictions.append(sine_model.get_tensor(sine_model_output_index)[0])

        # Same procedure to a quantized model
        sine_model_quantized.set_tensor(sine_model_quantized_input_index, x_value_tensor)
        sine_model_quantized.invoke()
        sine_model_quantized_predictions.append(sine_model_quantized.get_tensor(sine_model_quantized_output_index)[0])

# Confirm how data is aligned
plt.clf()
plt.title('Comparison of various models against actual values')
plt.plot(x_test, y_test, 'bo', label = 'Actual')
plt.plot(x_test, predictions, 'ro', label = 'Original predictions')
plt.plot(x_test, sine_model_predictions, 'bx', label = 'Lite predictions')
plt.plot(x_test, sine_model_quantized_predictions, 'gx', label = 'Lite quantized predictions')
plt.legend()
plt.savefig("Tf.lite_predictions.png")
plt.show()

#---------------------------------------Size Comparison--------------------------------------#

basic_model_size = os.path.getsize("sine_model.tflite")
print("Basic model is %d bytes" % basic_model_size)
quantized_model_size = os.path.getsize("sine_model_quantized.tflite")
print("Quantized model is %d bytes" % quantized_model_size)
difference = basic_model_size - quantized_model_size
print("Difference is %d bytes" % difference)