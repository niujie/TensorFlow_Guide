# TensorFlow Tensors
import tensorflow as tf

# Rank 0
mammal = tf.Variable("Elephant", tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.14159265359, tf.float64)
its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)

# Rank 1
mystr = tf.Variable(["Hello"], tf.string)
cool_numbers = tf.Variable([3.14159, 2.71828], tf.float32)
first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)

# Higher ranks
mymat = tf.Variable([[7], [11]], tf.int16)
myxor = tf.Variable([[False, True], [True, False]], tf.bool)
linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
squarish_squares = tf.Variable([[4, 9], [16, 25]], tf.int32)
rank_of_squares = tf.rank(squarish_squares)
mymatC = tf.Variable([[7], [11]], tf.int32)

my_image = tf.zeros([10, 299, 299, 3])  # batch x height x width x color

# Getting a tf.Tensor object's rank
r = tf.rank(my_image)
# After the graph runs, r will hold the value 4.

# Referring to tf.Tensor slices
# my_scalar = my_vector[2]
# my_scalar = my_matrix[1, 2]
# my_row_vector = my_matrix[2]
# my_column_vector = my_matrix[:, 3]

# Shape
# Getting a tf.Tensor object's shape
# zeros = tf.zeros(my_matrix.shape[1])

# Changing the shape of a tf.Tensor
rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6, 10])  # Reshape existing content into
# a 6x10 matrix
matrixB = tf.reshape(matrix, [3, -1])  # Reshape existing content into a 3x20
# matrix. -1 tells reshape to calculate
# the size of this dimension.
matrixAlt = tf.reshape(matrixB, [4, 3, -1])  # Reshape existing content into a
# 4x3x5 tensor

# Note that the number of elements of the reshaped Tensors has to match the
# original number of elements. Therefore, the following example generates an
# error because no possible value for the last dimension will match the number
# of elements.
# yet_another = tf.reshape(matrixAlt, [13, 2, -1])  # ERROR!

# Data types
# Cast a constant integer tensor into floating point.
float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)

# Evaluate tensors
constant = tf.constant([1, 2, 3])
tensor = constant * constant
# print(tensor.eval())  # The eval method only works when a default tf.Session is active

'''
p = tf.placeholder(tf.float32)
t = p + 1.0
t.eval()  # This will fail, since the placeholder did not get a value.
t.eval(feed_dict={p: 2.0})  # This will succeed because we're feeding a value to the placeholder.
'''
