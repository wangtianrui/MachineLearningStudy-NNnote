import tensorflow as tf
import numpy as np

# * 和 matmul

array2x3 = np.array([[1, 2, 3],
                     [4, 5, 6]], dtype=float)

array3x4 = np.array([[7, 8, 9, 10],
                     [11, 12, 13, 14],
                     [15, 16, 17, 18]], dtype=float)
print(array2x3)
print(array3x4)
#print(array2x3 * array3x4)     error
array1x5 = np.array([1, 2, 3, 4, 5])
array1x52 = np.array([6, 7, 8, 9, 10])
print(array1x5 * array1x52)

with tf.Session() as sess:
    print(sess.run(tf.matmul(array2x3, array3x4)))
