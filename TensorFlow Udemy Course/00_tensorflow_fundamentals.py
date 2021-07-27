import numpy as np
import tensorflow as tf

if tf.__version__ != '2.5.0':
    print('Warning: TensorFlow version is not 2.5.0. Please upgrade it via \'python -m pip install --upgrade tensorflow\'')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                                                     #
#                                                   Constant tensors                                                  #
#                                                                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

print("Constant tensors: ")

# Create a scalar:
scalar = tf.constant(7)
print(scalar)
print(scalar.ndim)  # 0

# Create a vector:
vector = tf.constant([7., 10.], dtype=tf.float16)
print(vector)
print(vector.ndim)  # 1

# Create a matrix:
matrix = tf.constant(
    [
        [7, 10],
        [10, 7]
    ],
    dtype=tf.int64)
print(matrix)
print(matrix.ndim)  # 2

# Create a tensor:
tensor = tf.constant([
    [
        [1, 2],
        [3, 4]
    ],
    [
        [5, 6],
        [7, 8]
    ]
])
print(tensor)
print(tensor.ndim)  # 3

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                                                     #
#                                                  Variable tensors                                                   #
#                                                                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

print("Variable tensors: ")

changeable_tensor = tf.Variable([10, 7])
print(changeable_tensor)
changeable_tensor[0].assign(14)
print(changeable_tensor)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                                                     #
#                                                   Random tensors                                                    #
#                                                                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

print("Radom tensors: ")

random_1 = tf.random.Generator.from_seed(42)
# create tensor from a normal distribution
random_1 = random_1.normal(shape=(3, 2))
print(random_1)

# shuffle a tensor
print('Original tensor:')
print(matrix)
print('Shuffled tensor:')
shuffled_matrix = tf.random.shuffle(matrix, seed=42)
print(shuffled_matrix)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                                                     #
#                                             Other ways to create tensors                                            #
#                                                                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# make a tensor full of ones
ones_tensor = tf.ones(shape=(3, 2))
print(ones_tensor)

# make a tensor full of zeros
zeros_tensor = tf.ones(shape=(2, 3))
print(zeros_tensor)

# create from a numpy array
numpy_array = np.arange(1, 25, dtype=np.int32)  # 24 elements: 1 to 25.
tensorflow_array = tf.constant(numpy_array, shape=[2, 4, 3])  # 2*4*3 = 24.
print(tensorflow_array)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                                                     #
#                                             Getting information tensors                                             #
#                                                                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# shape: the length (number of elements) of each of the dimensions of a tensor.
# rank: the number of tensor dimensions. A scalar has rank 0, a vector has rank 1, a matrix 3, and a tensor a rank n.
# axis or dimension: a particular dimension of a tensor.
# size: the total number of items in the tensor.

rank_4_tensor = tf.zeros([2, 3, 4, 5])
print(rank_4_tensor)

# Get various attributes of tensor
print(f"Datatype of every element: {rank_4_tensor.dtype}")
print(f"Number of dimensions (rank): {rank_4_tensor.ndim}")
print(f"Shape of tensor: {rank_4_tensor.shape}")
print(f"Elements along axis 0 of tensor: {rank_4_tensor.shape[0]}")
print(f"Elements along last axis of tensor: {rank_4_tensor.shape[-1]}")
# .numpy() converts to NumPy array
print(f"Total number of elements (2*3*4*5): {tf.size(rank_4_tensor).numpy()}")

# Get the first 2 items of each dimension
rank_4_tensor[:2, :2, :2, :2]

# Get the dimension from each index except for the final one
rank_4_tensor[:1, :1, :1, :]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                                                     #
#                                                 Manipulating tensors                                                #
#                                                                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Add an extra dimension (to the end). In Python "..." means "all dimensions prior to"
rank_5_tensor = rank_4_tensor[..., tf.newaxis]

# You can achieve the same using tf.expand_dims().
other_rank_5_tensor = tf.expand_dims(
    rank_4_tensor, axis=-1)  # "-1" means last axis

# addition. Since matrix is constant, we will get a copy.
print(matrix + 5)

# substraction. Since matrix is constant, we will get a copy.
print(matrix - 5)

# multiplication. Since matrix is constant, we will get a copy.
print(matrix * 2)
print(tf.multiply(matrix, 3))

# Matrix multiplication. Note: '@' in Python is the symbol for matrix multiplication.
print(matrix @ matrix)
print(tf.matmul(matrix, matrix))

# To manipulate the dimensions of a tensor we can use:
# tf.reshape() - allows us to reshape a tensor into a defined shape.
# tf.reshape(Y, shape=(2, 3))
# Try matrix multiplication with reshaped Y
# X @ tf.reshape(Y, shape=(2, 3))
# tf.transpose() - switches the dimensions of a given tensor.
# tf.transpose(X)
# You can achieve the same result with parameters
# tf.matmul(a=X, b=Y, transpose_a=True, transpose_b=False)

# dot product
print(tf.tensordot(vector, tf.transpose(vector), axes=1))

# changing the datatype of a tensor
vector = tf.cast(vector, dtype=tf.float32)

# getting the absolute value
negative_vector = tf.constant([-3, -12])
absolute_vector = tf.abs(negative_vector)

# You can quickly aggregate (perform a calculation on a whole tensor) tensors to find things like the minimum value,
#  maximum value, mean and sum of all the elements.

# To do so, aggregation methods typically have the syntax reduce()_[action], such as:

# tf.reduce_min() - find the minimum value in a tensor.
# tf.reduce_max() - find the maximum value in a tensor(helpful for when you want to find the highest
# prediction probability).
# tf.reduce_mean() - find the mean of all elements in a tensor.
# tf.reduce_sum() - find the sum of all elements in a tensor.
# Note: typically, each of these is under the math module, e.g. tf.math.reduce_min() but you can use
# the alias tf.reduce_min().

random_tensor = tf.constant(np.random.randint(low=0, high=100, size=50))
print(f'The minimum value of random_tensor: {tf.reduce_min(random_tensor)}')
print(f'The maximum value of random_tensor: {tf.reduce_max(random_tensor)}')
print(f'The mean of random_tensor: {tf.reduce_mean(random_tensor)}')
print(f'The sum of random_tensor: {tf.reduce_sum(random_tensor)}')

# positional minimum and maximum

# tf.argmax() - find the position of the maximum element in a given tensor.
# tf.argmin() - find the position of the minimum element in a given tensor.

print(
    f'The position of the minimum value of random_tensor: {tf.argmin(random_tensor)}')
print(
    f'The position of the maximum value of random_tensor: {tf.argmax(random_tensor)}')

# squeezing a tensor (removing all single dimensions)

tensor_to_squeeze = tf.constant(
    np.random.randint(0, 100, 50), shape=(1, 1, 1, 2, 25))
print(f'Shape before squeezing: {tensor_to_squeeze.shape}')
print(f'Dimensions before squeezing: {tensor_to_squeeze.ndim}')

tensor_squeezed = tf.squeeze(tensor_to_squeeze)
print(f'Shape after squeezing: {tensor_squeezed.shape}')
print(f'Dimensions after squeezing: {tensor_squeezed.ndim}')

# other math operations:
# tf.square() - get the square of every value in a tensor.
# tf.sqrt() - get the squareroot of every value in a tensor(note: the elements need to be floats or this will error).
# tf.math.log() - get the natural log of every value in a tensor(elements need to floats).

# manipulating tf.Variable tensors
# .assign() - assign a different value to a particular index of a variable tensor.
# .add_assign() - add to an existing value and reassign it at a particular index of a variable tensor.

# tensors can also be converted to NumPy arrays using:
# np.array() - pass a tensor to convert to an ndarray (NumPy's main datatype).
# tensor.numpy() - call on a tensor to convert to an ndarray.

# Using @tf.function
# In your TensorFlow adventures, you might come across Python functions which have the decorator @tf.function.
# In the @tf.function decorator case, it turns a Python function into a callable TensorFlow graph. Which is a
# fancy way of saying, if you've written your own Python function, and you decorate it with @tf.function,
# when you export your code (to potentially run on another device), TensorFlow will attempt to convert it into
# a fast(er) version of itself (by making it part of a computation graph).
