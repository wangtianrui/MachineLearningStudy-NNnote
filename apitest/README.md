#### python种 * 运算符 和tf.matmul()的区别

>  “ * ” 运算符是将两个等shape矩阵进行各元素一一对应地相乘，tf.matmul()则是进行矩阵乘法运算

```python
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
#print(array2x3 * array3x4)    error    
#*运算符只能用在两个矩阵shape相同的情况
array1x5 = np.array([1, 2, 3, 4, 5])
array1x52 = np.array([6, 7, 8, 9, 10])
print(array1x5 * array1x52)

with tf.Session() as sess:
    print(sess.run(tf.matmul(array2x3, array3x4)))
    
"""
E:\python3.5\python.exe E:/python_programes/MachineLearningStudy/apitest/one.py
[[ 1.  2.  3.]
 [ 4.  5.  6.]]
[[  7.   8.   9.  10.]
 [ 11.  12.  13.  14.]
 [ 15.  16.  17.  18.]]
[ 6 14 24 36 50]
2018-03-25 19:36:51.457460: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2018-03-25 19:36:52.134421: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1212] Found device 0 with properties: 
name: GeForce GTX 965M major: 5 minor: 2 memoryClockRate(GHz): 1.15
pciBusID: 0000:01:00.0
totalMemory: 2.00GiB freeMemory: 1.64GiB
2018-03-25 19:36:52.134725: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1312] Adding visible gpu devices: 0
2018-03-25 19:36:52.454771: I C:\tf_jenkins\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1405 MB memory) -> physical GPU (device: 0, name: GeForce GTX 965M, pci bus id: 0000:01:00.0, compute capability: 5.2)
[[  74.   80.   86.   92.]
 [ 173.  188.  203.  218.]]
"""

```

