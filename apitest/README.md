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

#### tf.nn.softmax_cross_entropy_with_logits测试

> 测试softmax和cross_entropy分开写  和 使用混合的api的区别    ， 测试结果：没区别
    顺便测试了 label 取int和float的区别：无

```python
def testSoftMaxAndCross_entropy():
    """
    测试softmax和cross_entropy分开写  和 使用混合的api的区别    ， 测试结果：没区别（batch_size = 1）
    顺便测试了 label 取int和float的区别：无（batch_size = 1）,

    这里顺便总结一下为什么有些地方喜欢在计算交叉熵的时候用readuce_mean，个人考虑是为了“降低学习速率”，求平均可以在考虑到所有label的情况下相对于sum值更小
    :return:
    """
    # label = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype="float")  # label为8
    label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype="int64")  # label为8
    # array = np.array([0, 1.1, 2.5, 3.34, 4.346436, 5.434, 6.46, 7.2352, 9.234, 10.234], dtype="float")
    array = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
                      [1, 0, 3, 5, 7, 1, 2, 4, 6, 12]], dtype="float")
    mysoftmax = tf.nn.softmax(array)
    mycross = tf.reduce_sum(label * tf.log(mysoftmax))
    apicross = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=array, labels=label))
    with tf.Session() as sess:
        print(sess.run(mysoftmax))
        print("------------------------------------------------------------")
        print(sess.run(mycross))
        print("------------------------------------------------------------")  # label int 1.36922634586
        print(sess.run(apicross))
"""
[[  3.13835607e-05   8.53093628e-05   2.31894891e-04   6.30355668e-04
    1.71348436e-03   4.65773339e-03   1.26610320e-02   3.44162533e-02
    2.54303627e-01   6.91268927e-01]
 [  1.65253358e-05   6.07933129e-06   1.22106633e-04   9.02252762e-04
    6.66679628e-03   1.65253358e-05   4.49205200e-05   3.31920242e-04
    2.45257729e-03   9.89440296e-01]]
------------------------------------------------------------
-13.3798421989
------------------------------------------------------------
13.3798421989
"""


```
