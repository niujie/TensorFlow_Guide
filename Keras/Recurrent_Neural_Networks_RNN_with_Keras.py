# Setup
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from tensorflow.keras import layers

# 获取所有GPU设备列表
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU显存占用为按需分配
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUS,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # 异常处理
        print(e)

# Build a simple model
model = tf.keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))

# Add a Dense layer with 10 units and softmax activation.
model.add(layers.Dense(10, activation='softmax'))

model.summary()
# LSTM Param #: 4 * [(128 + 64) * 128 + 128]

# Outputs and states
encoder_vocab = 1000
decoder_vocab = 2000

encoder_input = layers.Input(shape=(None,))
encoder_embedded = layers.Embedding(input_dim=encoder_vocab, output_dim=64)(encoder_input)

# Return states in addition to output
output, state_h, state_c = layers.LSTM(
    64, return_state=True, name='encoder')(encoder_embedded)
encoder_state = [state_h, state_c]

decoder_input = layers.Input(shape=(None,))
decoder_embedded = layers.Embedding(input_dim=decoder_vocab, output_dim=64)(decoder_input)

# Pass the 2 states to a new LSTM layer, as initial state
decoder_output = layers.LSTM(
    64, name='decoder')(decoder_embedded, initial_state=encoder_state)
output = layers.Dense(10, activation='softmax')(decoder_output)

model = tf.keras.Model([encoder_input, decoder_input], output)
model.summary()
# LSTM Param #: 4 * [(64 + 64) * 64 + 64]

# Cross-batch statefulness
paragraph1 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph2 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph3 = np.random.random((20, 10, 50)).astype(np.float32)

lstm_layer = layers.LSTM(64, stateful=True)
output = lstm_layer(paragraph1)
output = lstm_layer(paragraph2)
output = lstm_layer(paragraph3)

# reset_states() will reset the cached state to the original initial_state.
# If no initial_state was provided, zero-states will be used by default.
lstm_layer.reset_states()

# Bidrectional RNNs
model = tf.keras.Sequential()

model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True),
                               input_shape=(5, 10)))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(10, activation='softmax'))

model.summary()
# Bidirectional LSTM layer 1 Param #: 2 * {4 * [(10 + 64) * 64 + 64]}
# Bidirectional LSTM layer 2 Param #: 2 * {4 * [(128 + 32) * 32 + 32]}

# Performance optimization and CuDNN kernels in TensorFlow 2.0
# Using CuDNN kernels when available
batch_size = 64
# Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
# Each input sequence will be of size (28, 28) (height is treated like time).
input_dim = 28

units = 64
output_size = 10  # labels are from 0 to 9


# Build the RNN model
def build_model(allow_cudnn_kernel=True):
    # CuDNN is only available at the layer level, and not at the cell level.
    # This means `LSTM(units)` will use the CuDNN kernel,
    # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
    if allow_cudnn_kernel:
        # The LSTM layer with default options uses CuDNN.
        lstm_layer_ = tf.keras.layers.LSTM(units, input_shape=(None, input_dim))
    else:
        # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
        lstm_layer_ = tf.keras.layers.RNN(
            tf.keras.layers.LSTMCell(units),
            input_shape=(None, input_dim))
    model_ = tf.keras.models.Sequential([
        lstm_layer_,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(output_size, activation='softmax')]
    )
    return model_


# Load MNIST dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
sample, sample_label = x_train[0], y_train[0]

# Create a model instance and compile it
model = build_model(allow_cudnn_kernel=True)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=batch_size,
          epochs=5)

# Build a new model without CuDNN kernel
slow_model = build_model(allow_cudnn_kernel=False)
slow_model.set_weights(model.get_weights())
slow_model.compile(loss='sparse_categorical_crossentropy',
                   optimizer='sgd',
                   metrics=['accuracy'])
slow_model.fit(x_train, y_train,
               validation_data=(x_test, y_test),
               batch_size=batch_size,
               epochs=1)  # We only train for one epoch because it's slower.

with tf.device('CPU:0'):
    cpu_model = build_model(allow_cudnn_kernel=True)
    cpu_model.set_weights(model.get_weights())
    result = tf.argmax(cpu_model.predict_on_batch(tf.expand_dims(sample, 0)), axis=1)
    print('Predicted result is: %s, target result is: %s' % (result.numpy(), sample_label))
    plt.imshow(sample, cmap=plt.get_cmap('gray'))
    plt.show()

# RNNs with list/dict inputs, or nested inputs
# Define a custom cell that support nested input/output
NestedInput = collections.namedtuple('NestedInput', ['feature1', 'feature2'])
NestedState = collections.namedtuple('NestedState', ['state1', 'state2'])


class NestedCell(tf.keras.layers.Layer):

    def __init__(self, unit_1_, unit_2_, unit_3_, **kwargs):
        self.unit_1 = unit_1_
        self.unit_2 = unit_2_
        self.unit_3 = unit_3_
        self.state_size = NestedState(state1=unit_1_,
                                      state2=tf.TensorShape([unit_2_, unit_3_]))
        self.output_size = (unit_1_, tf.TensorShape([unit_2_, unit_3_]))
        super(NestedCell, self).__init__(**kwargs)

    def build(self, input_shapes):
        # expect input_shape to contain 2 items, [(batch, i1), (batch, i2, i3)]
        input_1_ = input_shapes.feature1[1]
        input_2_, input_3_ = input_shapes.feature2[1:]

        self.kernel_1 = self.add_weight(
            shape=(input_1_, self.unit_1), initializer='uniform', name='kernel_1')
        self.kernel_2_3 = self.add_weight(
            shape=(input_2_, input_3_, self.unit_2, self.unit_3),
            initializer='uniform',
            name='kernel_2_3')

    def call(self, inputs, states):
        # inputs should be in [(batch, input_1), (batch, input_2, input_3)]
        # state should be in shape [(batch, unit_1), (batch, unit_2, unit_3)]
        input_1_, input_2_ = tf.nest.flatten(inputs)
        s1, s2 = states

        output_1 = tf.matmul(input_1_, self.kernel_1)
        output_2_3 = tf.einsum('bij,ijkl->bkl', input_2_, self.kernel_2_3)
        state_1 = s1 + output_1
        state_2_3 = s2 + output_2_3

        output_ = [output_1, output_2_3]
        new_states = NestedState(state1=state_1, state2=state_2_3)

        return output_, new_states


# Build a RNN model with nested input/output
unit_1 = 10
unit_2 = 20
unit_3 = 30

input_1 = 32
input_2 = 64
input_3 = 32
batch_size = 64
num_batch = 100
timestep = 50

cell = NestedCell(unit_1, unit_2, unit_3)
rnn = tf.keras.layers.RNN(cell)

inp_1 = tf.keras.Input((None, input_1))
inp_2 = tf.keras.Input((None, input_2, input_3))

outputs = rnn(NestedInput(feature1=inp_1, feature2=inp_2))

model = tf.keras.models.Model([inp_1, inp_2], outputs)

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train the model with randomly generated data
input_1_data = np.random.random((batch_size * num_batch, timestep, input_1))
input_2_data = np.random.random((batch_size * num_batch, timestep, input_2, input_3))
target_1_data = np.random.random((batch_size * num_batch, unit_1))
target_2_data = np.random.random((batch_size * num_batch, unit_2, unit_3))
input_data = [input_1_data, input_2_data]
target_data = [target_1_data, target_2_data]

model.fit(input_data, target_data, batch_size=batch_size)
