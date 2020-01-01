# Better performance with tf.function and AutoGraph

# Setup
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf


# The tf.function decorator
@tf.function
def simple_nn_layer(x_, y_):
    return tf.nn.relu(tf.matmul(x_, y_))


x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

print(simple_nn_layer(x, y))
print(simple_nn_layer)


def linear_layer(x_):
    return 2 * x_ + 1


@tf.function
def deep_net(x_):
    return tf.nn.relu(linear_layer(x_))


print(deep_net(tf.constant((1, 2, 3))))

import timeit

conv_layer = tf.keras.layers.Conv2D(100, 3)


@tf.function
def conv_fn(image_):
    return conv_layer(image_)


image = tf.zeros([1, 200, 200, 100])
# warm up
conv_layer(image)
conv_fn(image)
print("Eager conv:", timeit.timeit(lambda: conv_layer(image), number=10))
print("Function conv:", timeit.timeit(lambda: conv_fn(image), number=10))
print("Note how there's not much difference in performance for convolutions")

lstm_cell = tf.keras.layers.LSTMCell(10)


@tf.function
def lstm_fn(_input_, state_):
    return lstm_cell(_input_, state_)


input_ = tf.zeros([10, 10])
state = [tf.zeros([10, 10])] * 2
# warm up
lstm_cell(input_, state)
lstm_fn(input_, state)
print("eager lstm:", timeit.timeit(lambda: lstm_cell(input_, state), number=10))
print("function lstm:", timeit.timeit(lambda: lstm_fn(input_, state), number=10))


# Use Python control flow
@tf.function
def square_if_positive(x_):
    if x_ > 0:
        x_ = x_ * x_
    else:
        x_ = 0
    return x_


print('square_if_positive(2) = {}'.format(square_if_positive(tf.constant(2))))
print('square_if_positive(-2) = {}'.format(square_if_positive(tf.constant(-2))))


@tf.function
def sum_even(items):
    s = 0
    for c in items:
        if c % 2 > 0:
            continue
        s += c
    return s


print(sum_even(tf.constant([10, 12, 15, 20])))

print(tf.autograph.to_code(sum_even.python_function))


@tf.function
def fizzbuzz(n):
    for i in tf.range(n):
        if i % 3 == 0:
            tf.print('Fizz')
        elif i % 5 == 0:
            tf.print('Buzz')
        else:
            tf.print(i)


fizzbuzz(tf.constant(15))


# Keras and AutoGraph
class CustomModel(tf.keras.models.Model):

    @tf.function
    def call(self, input_data):
        if tf.reduce_mean(input_data) > 0:
            return input_data
        else:
            return input_data // 2


model = CustomModel()

print(model(tf.constant([-2, -4])))

# Side effects
v = tf.Variable(5)


@tf.function
def find_next_odd():
    v.assign(v + 1)
    if v % 2 == 0:
        v.assign(v + 1)


find_next_odd()
print(v)


# Debugging
@tf.function
def f(x_):
    if x_ > 0:
        # Try setting a breakpoint here!
        # Example:
        #   import pdb
        #   pdb.set_trace()
        x_ = x_ + 1
    return x_


tf.config.experimental_run_functions_eagerly(True)

# You can now set breakpoints and run the code in a debugger.
f(tf.constant(1))

tf.config.experimental_run_functions_eagerly(False)


# Advanced example: An in-graph training loop
# Download data
def prepare_mnist_features_and_labels(x_, y_):
    x_ = tf.cast(x_, tf.float32) / 255.0
    y_ = tf.cast(y_, tf.int64)
    return x_, y_


def mnist_dataset():
    (x_, y_), _ = tf.keras.datasets.mnist.load_data()
    ds = tf.data.Dataset.from_tensor_slices((x_, y_))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(100)
    return ds


train_dataset = mnist_dataset()

# Define the model
model = tf.keras.Sequential((
    tf.keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10)))
model.build()
model.summary()
optimizer = tf.keras.optimizers.Adam()

# Define the training loop
compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


def train_one_step(model_, optimizer_, x_, y_):
    with tf.GradientTape() as tape:
        logits = model_(x_)
        loss_ = compute_loss(y_, logits)

    grads = tape.gradient(loss_, model_.trainable_variables)
    optimizer_.apply_gradients(zip(grads, model_.trainable_variables))

    compute_accuracy(y_, logits)
    return loss_


@tf.function
def train(model_, optimizer_):
    train_ds = mnist_dataset()
    step_ = 0
    loss_ = 0.0
    accuracy_ = 0.0
    for x_, y_ in train_ds:
        step_ += 1
        loss_ = train_one_step(model_, optimizer_, x_, y_)
        if step_ % 10 == 0:
            tf.print('Step', step_, ': loss', loss_, '; accuracy', compute_accuracy.result())
    return step_, loss_, accuracy_


step, loss, accuracy = train(model, optimizer)
print('Final step', step, ': loss', loss, '; accuracy', compute_accuracy.result())


# Batching
def square_if_positive(x_):
    return [i ** 2 if i > 0 else i for i in x_]


print(square_if_positive(range(-5, 5)))


@tf.function
def square_if_positive_naive(x_):
    result = tf.TensorArray(tf.int32, size=x_.shape[0])
    for i in tf.range(x_.shape[0]):
        if x_[i] > 0:
            result = result.write(i, x_[i] ** 2)
        else:
            result = result.write(i, x_[i])
    return result.stack()


print(square_if_positive_naive(tf.range(-5, 5)))


def square_if_positive_vectorized(x_):
    return tf.where(x_ > 0, x_ ** 2, x_)


print(square_if_positive_vectorized(tf.range(-5, 5)))

# Retracing
import timeit


@tf.function
def f(x_, y_):
    return tf.matmul(x_, y_)


print(
    "First invocation:",
    timeit.timeit(lambda: f(tf.ones((10, 10)), tf.ones((10, 10))), number=1))

print(
    "Second invocation:",
    timeit.timeit(lambda: f(tf.ones((10, 10)), tf.ones((10, 10))), number=1))


@tf.function
def f():
    print('Tracing!')
    tf.print('Executing')


print('First invocation:')
f()

print('Second invocation:')
f()


@tf.function
def f(n):
    print(n, 'Tracing!')
    tf.print(n, 'Executing')


f(1)
f(1)

f(2)
f(2)


@tf.function
def f(x_):
    print(x_.shape, 'Tracing!')
    tf.print(x_, 'Executing')


f(tf.constant([1]))
f(tf.constant([2]))

f(tf.constant([1, 2]))
f(tf.constant([3, 4]))


def f():
    print('Tracing!')
    tf.print('Executing')


tf.function(f)()
tf.function(f)()


def outer():
    @tf.function
    def f_():
        print('Tracing!')
        tf.print('Executing')

    f_()


outer()
outer()
