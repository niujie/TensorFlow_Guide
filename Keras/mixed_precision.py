# Setup
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# Setting the dtype policy
# policy = mixed_precision.Policy('mixed_float16')
policy = mixed_precision.Policy('float32')
mixed_precision.set_policy(policy)

print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

# Building the model
inputs = keras.Input(shape=(784,), name='digits')
if tf.config.experimental.list_physical_devices('GPU'):
    print('The model will run with 4096 units on a GPU')
    num_units = 4096
else:
    # Use fewer units on CPUs so the model finishes in a reasonable amount of time
    print('The model will run with 64 units on a CPU')
    num_units = 64
dense1 = layers.Dense(num_units, activation='relu', name='dense_1')
x = dense1(inputs)
dense2 = layers.Dense(num_units, activation='relu', name='dense_2')
x = dense2(x)

print('x.dtype: %s' % x.dtype.name)
# 'kernel' is dense1's variable
print('dense1.kernel.dtype: %s' % dense1.kernel.dtype.name)

# INCORRECT: softmax and model output will be float16, when it should be float32
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
print('Outputs dtype: %s' % outputs.dtype.name)

# CORRECT: softmax and model output are float32
x = layers.Dense(10, name='dense_logits')(x)
outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x)
print('Outputs dtype: %s' % outputs.dtype.name)

# The linear activation is an identity function. So this simply casts 'outputs'
# to float32. In this particular case, 'outputs' is already float32 so this is a
# no-op.
outputs = layers.Activation('linear', dtype='float32')(outputs)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

initial_weights = model.get_weights()

# Training the model with Model.fit
history = model.fit(x_train, y_train,
                    batch_size=8192,
                    epochs=5,
                    validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

# Loss scaling
# Underflow and Overflow
x = tf.constant(256, dtype='float16')
print((x ** 2).numpy())  # Overflow

x = tf.constant(1e-5, dtype='float16')
print((x ** 2).numpy())  # Underflow

# Loss scaling background
'''
loss_scale = 1024
loss = model(inputs)
loss *= loss_scale
# We assume `grads` are float32. We do not want to divide float16 gradients
grads = compute_gradient(loss, model.trainable_variables)
grads /= loss_scale
'''

# Choosing the loss scale
loss_scale = policy.loss_scale
print('Loss scale: %s' % loss_scale)

new_policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
print(new_policy.loss_scale)

# Training the model with a custom training loop
optimizer = keras.optimizers.RMSprop()
optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                 .shuffle(10000).batch(8192))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(8192)


@tf.function
def train_step(x_, y_):
    with tf.GradientTape() as tape:
        predictions_ = model(x_)
        loss_ = loss_object(y_, predictions_)
        scaled_loss = optimizer.get_scaled_loss(loss_)
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_


@tf.function
def test_step(x_):
    return model(x_, training=False)


model.set_weights(initial_weights)

for epoch in range(5):
    epoch_loss_avg = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')
    for x, y in train_dataset:
        loss = train_step(x, y)
        epoch_loss_avg(loss)
    for x, y in test_dataset:
        predictions = test_step(x)
        test_accuracy.update_state(y, predictions)
    print('Epoch {}: loss={}, test accuracy={}'.format(epoch, epoch_loss_avg.result(), test_accuracy.result()))
