# TensorFlow variables
import tensorflow as tf

# Create a variable
my_variable = tf.Variable(tf.zeros([1., 2., 3.]))

if tf.config.experimental.list_physical_devices('GPU'):
    with tf.device("/device:GPU:1"):
        v = tf.Variable(tf.zeros([10, 10]))

# Use a variable
v = tf.Variable(0.0)
w = v + 1  # w is a tf.Tensor which is computed based on the value of v.
# Any time a variable is used in an expression it gets automatically
# converted to a tf.Tensor representing its value.

v = tf.Variable(0.0)
v.assign_add(1)

v = tf.Variable(0.0)
v.assign_add(1)
v.read_value()  # 1.0


# Keep track of variables
class MyLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(MyLayer, self).__init__()
        self.my_var = tf.Variable(1.0)
        self.my_var_list = [tf.Variable(x) for x in range(10)]


class MyOtherLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(MyOtherLayer, self).__init__()
        self.sublayer = MyLayer()
        self.my_other_var = tf.Variable(10.0)


m = MyOtherLayer()
print(len(m.variables))  # 12 (11 from MyLayer, plus my_other_var)
