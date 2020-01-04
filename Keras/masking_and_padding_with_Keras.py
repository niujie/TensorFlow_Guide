# Setup
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf

from tensorflow.keras import layers

# Padding sequence data
'''
[
  ["The", "weather", "will", "be", "nice", "tomorrow"],
  ["How", "are", "you", "doing", "today"],
  ["Hello", "world", "!"]
]
[
  [83, 91, 1, 645, 1253, 927],
  [73, 8, 3215, 55, 927],
  [71, 1331, 4231]
]
'''
raw_inputs = [
    [83, 91, 1, 645, 1253, 927],
    [73, 8, 3215, 55, 927],
    [711, 632, 71]
]

# By default, this will pad using 0s; it is configurable via the
# "value" parameter.
# Note that you could "pre" padding (at the beginning) or
# "post" padding (at the end).
# We recommend using "post" padding when working with RNN layers
# (in order to be able to use the
# CuDNN implementation of the layers).
padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs,
                                                              padding='post')

print(padded_inputs)

# Masking
# Mask-generating layers: Embedding and Masking
embedding = layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)
masked_output = embedding(padded_inputs)

print(masked_output._keras_mask)

masking_layer = layers.Masking()
# Simulate the embedding lookup by expanding the 2D input to 3D,
# with embedding dimension of 10.
unmasked_embedding = tf.cast(
    tf.tile(tf.expand_dims(padded_inputs, axis=-1), [1, 1, 10]),
    tf.float32)

masked_embedding = masking_layer(unmasked_embedding)
print(masked_embedding._keras_mask)

# Mask propagation in the Functional API and Sequential API
model = tf.keras.Sequential([
    layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True),
    layers.LSTM(32),
])

inputs = tf.keras.Input(shape=(None,), dtype='int32')
x = layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)(inputs)
outputs = layers.LSTM(32)(x)

model = tf.keras.Model(inputs, outputs)


# Passing mask tensors directly to layers
class MyLayer(layers.Layer):

    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.embedding = layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)
        self.lstm = layers.LSTM(32)

    def call(self, inputs_):
        x_ = self.embedding(inputs_)
        # Note that you could also prepare a `mask` tensor manually.
        # It only needs to be a boolean tensor
        # with the right shape, i.e. (batch_size, timesteps).
        mask_ = self.embedding.compute_mask(inputs_)
        output = self.lstm(x_, mask=mask_)  # The layer will ignore the masked values
        return output


layer = MyLayer()
x = np.random.random((32, 10)) * 100
x = x.astype('int32')
print(layer(x))


# Supporting masking in your custom layers
class TemporalSplit(tf.keras.layers.Layer):
    """Split the input tensor into 2 tensors along the time dimension."""

    def call(self, inputs_):
        # Expect the input to be 3D and mask to be 2D, split the input tensor into 2
        # subtensors along the time axis (axis 1).
        return tf.split(inputs_, 2, axis=1)

    def compute_mask(self, inputs_, mask_=None):
        # Also split the mask into 2 if it presents.
        if mask_ is None:
            return None
        return tf.split(mask_, 2, axis=1)


first_half, second_half = TemporalSplit()(masked_embedding)
print(first_half._keras_mask)
print(second_half._keras_mask)


class CustomEmbedding(tf.keras.layers.Layer):

    def __init__(self, input_dim, output_dim, mask_zero=False, **kwargs):
        super(CustomEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer='random_normal',
            dtype='float32')

    def call(self, inputs_):
        return tf.nn.embedding_lookup(self.embeddings, inputs_)

    def compute_mask(self, inputs_, mask_=None):
        if not self.mask_zero:
            return None
        return tf.not_equal(inputs_, 0)


layer = CustomEmbedding(10, 32, mask_zero=True)
x = np.random.random((3, 10)) * 9
x = x.astype('int32')

y = layer(x)
mask = layer.compute_mask(x)

print(mask)
