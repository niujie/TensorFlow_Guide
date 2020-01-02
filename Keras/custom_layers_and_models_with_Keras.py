# Writing custom layers and models with Keras
# Setup
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf.keras.backend.clear_session()  # For easy reset of notebook state.

# The Layer class
from tensorflow.keras import layers


class Linear(layers.Layer):

    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                                  dtype='float32'),
                             trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(units,),
                                                  dtype='float32'),
                             trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)

assert linear_layer.weights == [linear_layer.w, linear_layer.b]


class Linear(layers.Layer):

    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)


class ComputeSum(layers.Layer):

    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)),
                                 trainable=False)

    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total


x = tf.ones((2, 2))
my_sum = ComputeSum(2)
y = my_sum(x)
print(y.numpy())
y = my_sum(x)
print(y.numpy())

print('weights:', len(my_sum.weights))
print('non-trainable weights:', len(my_sum.non_trainable_weights))

# It's not included in the trainable weights:
print('trainable_weights:', my_sum.trainable_weights)


class Linear(layers.Layer):

    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(units,),
                                 initializer='zeros',
                                 trainable=True)


class Linear(layers.Layer):

    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


linear_layer = Linear(32)  # At instantiation, we don't know on what inputs this is going to get called
y = linear_layer(x)  # The layer's weights are created dynamically the first time the layer is called


# Let's assume we are reusing the Linear class
# with a `build` method that we defined above.

class MLPBlock(layers.Layer):

    def __init__(self):
        super(MLPBlock, self).__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(1)

    def call(self, inputs):
        x_ = self.linear_1(inputs)
        x_ = tf.nn.relu(x_)
        x_ = self.linear_2(x_)
        x_ = tf.nn.relu(x_)
        return self.linear_3(x_)


mlp = MLPBlock()
y = mlp(tf.ones(shape=(3, 64)))  # The first call to the `mlp` will create the weights
print('weights:', len(mlp.weights))
print('trainable weights:', len(mlp.trainable_weights))


# A layer that creates an activity regularization loss
class ActivityRegularizationLayer(layers.Layer):

    def __init__(self, rate=1e-2):
        super(ActivityRegularizationLayer, self).__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs


class OuterLayer(layers.Layer):

    def __init__(self):
        super(OuterLayer, self).__init__()
        self.activity_reg = ActivityRegularizationLayer(1e-2)

    def call(self, inputs):
        return self.activity_reg(inputs)


layer = OuterLayer()
assert len(layer.losses) == 0  # No losses yet since the layer has never been called
_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1  # We created one loss value

# `layer.losses` gets reset at the start of each __call__
_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1  # This is the loss created during the call above


class OuterLayer(layers.Layer):

    def __init__(self):
        super(OuterLayer, self).__init__()
        self.dense = layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    def call(self, inputs):
        return self.dense(inputs)


layer = OuterLayer()
_ = layer(tf.zeros((1, 1)))

# This is `1e-3 * sum(layer.dense.kernel ** 2)`,
# created by the `kernel_regularizer` above.
print(layer.losses)

'''
# Instantiate an optimizer.
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Iterate over the batches of a dataset.
for x_batch_train, y_batch_train in train_dataset:
    with tf.GradientTape() as tape:
        logits = layer(x_batch_train)  # Logits for this minibatch
        # Loss value for this minibatch
        loss_value = loss_fn(y_batch_train, logits)
        # Add extra losses created during this forward pass:
        loss_value += sum(model.losses)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
'''


class Linear(layers.Layer):

    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {'units': self.units}


# Now you can recreate the layer from its config:
layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)


class Linear(layers.Layer):

    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config_ = super(Linear, self).get_config()
        config_.update({'units': self.units})
        return config_


layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)


# Privileged training argument in the call method
class CustomDropout(layers.Layer):

    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs


# Building Models
# The Model class
'''
class ResNet(tf.keras.Model):

    def __init__(self):
        super(ResNet, self).__init__()
        self.block_1 = ResNetBlock()
        self.block_2 = ResNetBlock()
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = Dense(num_classes)

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.global_pool(x)
        return self.classifier(x)


resnet = ResNet()
dataset = ...
resnet.fit(dataset, epochs=10)
resnet.save_weights(filepath)
'''


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean_, z_log_var_ = inputs
        batch = tf.shape(z_mean_)[0]
        dim = tf.shape(z_mean_)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean_ + tf.exp(0.5 * z_log_var_) * epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self,
                 latent_dim_=32,
                 intermediate_dim_=64,
                 name='encoder',
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim_, activation='relu')
        self.dense_mean = layers.Dense(latent_dim_)
        self.dense_log_var = layers.Dense(latent_dim_)
        self.sampling = Sampling()

    def call(self, inputs):
        x_ = self.dense_proj(inputs)
        z_mean_ = self.dense_mean(x_)
        z_log_var_ = self.dense_log_var(x_)
        z_ = self.sampling((z_mean_, z_log_var_))
        return z_mean_, z_log_var_, z_


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self,
                 original_dim_,
                 intermediate_dim_=64,
                 name='decoder',
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim_, activation='relu')
        self.dense_output = layers.Dense(original_dim_, activation='sigmoid')

    def call(self, inputs):
        x_ = self.dense_proj(inputs)
        return self.dense_output(x_)


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
                 original_dim_,
                 intermediate_dim_=64,
                 latent_dim_=32,
                 name='autoencoder',
                 **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim_
        self.encoder = Encoder(latent_dim_=latent_dim_,
                               intermediate_dim_=intermediate_dim_)
        self.decoder = Decoder(original_dim_, intermediate_dim_=intermediate_dim_)

    def call(self, inputs):
        z_mean_, z_log_var_, z_ = self.encoder(inputs)
        reconstructed_ = self.decoder(z_)
        # Add KL divergence regularization loss.
        kl_loss_ = - 0.5 * tf.reduce_mean(
            z_log_var_ - tf.square(z_mean_) - tf.exp(z_log_var_) + 1)
        self.add_loss(kl_loss_)
        return reconstructed_


original_dim = 784
vae = VariationalAutoEncoder(original_dim, 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

loss_metric = tf.keras.metrics.Mean()

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

epochs = 3

# Iterate over epochs.
for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))

    # Iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed = vae(x_batch_train)
            # Compute reconstruction loss
            loss = mse_loss_fn(x_batch_train, reconstructed)
            loss += sum(vae.losses)  # Add KLD regularization loss

        grads = tape.gradient(loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))

        loss_metric(loss)

        if step % 100 == 0:
            print('step %s: mean loss = %s' % (step, loss_metric.result()))

vae = VariationalAutoEncoder(784, 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=3, batch_size=64)

# Beyond object-oriented development: the Functional API
original_dim = 784
intermediate_dim = 64
latent_dim = 32

# Define encoder model.
original_inputs = tf.keras.Input(shape=(original_dim,), name='encoder_input')
x = layers.Dense(intermediate_dim, activation='relu')(original_inputs)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
z = Sampling()((z_mean, z_log_var))
encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name='encoder')

# Define decoder model.
latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = layers.Dense(original_dim, activation='sigmoid')(x)
decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name='decoder')

# Define VAE model.
outputs = decoder(z)
vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name='vae')

# Add KL divergence regularization loss.
kl_loss = - 0.5 * tf.reduce_mean(
    z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae.add_loss(kl_loss)

# Train.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=3, batch_size=64)
