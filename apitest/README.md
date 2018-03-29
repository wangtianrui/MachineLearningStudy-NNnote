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

### tf.nn.lrn

>    lrn操作通过调小“极差”（也就是将影响作用非常大的特征的数值进行调小）的操作，来防止过拟合的发生

```python
def testLRN():
    """
    LRN函数测试，相关资料查看NNnotes,整体来说lrn操作是将同块（相同区域但是不同channel的值进行中和的一个操作）
    注意：input须为float16, bfloat16, float32
    参数注意：
    调整beta=》beta则是调整k与alpha整体作用的参数,只要k大于1，永远是大于1的值，所以是调小
     alpha=》调整极差（越小极差越大,但是极差值不会超过原本输入值）
    个人理解 k（biase）值的设置是为了确定是让原本值调大调小的参数，一般设为1，这里讲一下为什么设为1，个人理解
    是如果k为1，那么通过调整alpha和beta值就可以来对整体“极差”进行调整（alpha值很小，那么“处理”的作用就会比较小,如果
    alpha比较大那么就会将整体数值进行调小），
    综上可大致得出结论：lrn操作通过调小“极差”（也就是将影响作用非常大的特征的数值进行调小）的操作，来防止过拟合的
    发生

    """

    array = np.array([[
        [1, 2, 3, 4, 5, 6, 7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16]],
        [[17, 18, 19, 20, 21, 22, 23, 24],
         [27, 28, 29, 30, 31, 32, 33, 34]]], dtype="float32")


    array = tf.reshape(array, shape=[1, 2, 8, 2])
    lrn_result = tf.nn.lrn(array, depth_radius=4, beta=0.5, bias=0, alpha=1)
    lrn_result1 = tf.nn.lrn(array, depth_radius=4, beta=1, bias=0, alpha=1)
    lrn_result2 = tf.nn.lrn(array, depth_radius=4, beta=0.05, bias=0, alpha=0.05)
    lrn_result3 = tf.nn.lrn(array, depth_radius=4, beta=0.5, bias=1, alpha=1)
    lrn_good = tf.nn.lrn(array, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    with tf.Session() as sess:
        print("------------------------------beta0.5,bias0,alpha1------------------------------")
        print(sess.run(lrn_result))
        print("-------------------------------beta1,bias0,alpha1-----------------------------")
        print(sess.run(lrn_result1))
        print("-------------------------------beta0.05,bias0,alpha0.05---------------------------")
        print(sess.run(lrn_result2))
        print("-------------------------------beta0.5,bias1,alpha1-----------------------------")
        print(sess.run(lrn_result3))
        print("------------------------------good parameter------------------------------")
        print(sess.run(lrn_good))
        
        
"""
------------------------------beta0.5,bias0,alpha1------------------------------
[[[[ 0.44721359  0.89442718]
   [ 0.60000002  0.80000001]
   [ 0.6401844   0.76822132]
   [ 0.65850455  0.75257665]
   [ 0.66896474  0.74329412]
   [ 0.67572463  0.73715413]
   [ 0.68045104  0.73279345]
   [ 0.68394119  0.72953725]]

  [[ 0.68662345  0.72701311]
   [ 0.68874949  0.72499943]
   [ 0.6904757   0.72335553]
   [ 0.69190526  0.72198814]
   [ 0.69413561  0.71984434]
   [ 0.69502211  0.71898836]
   [ 0.69579524  0.71824026]
   [ 0.69647539  0.71758074]]]]
-------------------------------beta1,bias0,alpha1-----------------------------
[[[[ 0.2         0.40000001]
   [ 0.12        0.16      ]
   [ 0.0819672   0.09836065]
   [ 0.0619469   0.07079646]
   [ 0.04972376  0.05524862]
   [ 0.04150943  0.04528302]
   [ 0.03561644  0.03835617]
   [ 0.03118503  0.03326403]]

  [[ 0.02773246  0.02936378]
   [ 0.02496715  0.02628121]
   [ 0.0227027   0.02378378]
   [ 0.02081448  0.02171946]
   [ 0.01784534  0.01850628]
   [ 0.01665709  0.01723148]
   [ 0.01561713  0.01612091]
   [ 0.01469933  0.01514477]]]]
-------------------------------beta0.05,bias0,alpha0.05-----------------------------
[[[[  1.07177341   2.14354682]
   [  2.96671438   3.95561934]
   [  4.72884417   5.674613  ]
   [  6.41941738   7.33647728]
   [  8.06139278   8.95710278]
   [  9.66678143  10.54557991]
   [ 11.24294853  12.10778999]
   [ 12.79485893  13.64785004]]

  [[ 14.32608032  15.16879082]
   [ 15.83929539  16.67294312]
   [ 17.33659172  18.16214371]
   [ 18.81963539  19.63788223]
   [ 21.74819946  22.55368805]
   [ 23.19581032  23.9956665 ]
   [ 24.63344574  25.42807388]
   [ 26.06181145  26.85156441]]]]
-------------------------------beta0.5,bias1,alpha1-----------------------------
[[[[ 0.40824828  0.81649655]
   [ 0.58834833  0.78446442]
   [ 0.63500059  0.76200074]
   [ 0.65561008  0.74926865]
   [ 0.66712433  0.74124926]
   [ 0.6744532   0.73576713]
   [ 0.67952085  0.73179168]
   [ 0.68323123  0.72877997]]

  [[ 0.68606418  0.72642094]
   [ 0.68829727  0.72452343]
   [ 0.69010282  0.72296482]
   [ 0.69159245  0.72166169]
   [ 0.69390619  0.71960646]
   [ 0.69482261  0.71878201]
   [ 0.69562     0.71805936]
   [ 0.69632024  0.71742082]]]]
------------------------------good parameter------------------------------
[[[[  0.99958354   1.99916708]
   [  2.99376512   3.99168682]
   [  4.97473335   5.96967983]
   [  6.93479919   7.92548466]
   [  8.86659527   9.85177231]
   [ 10.76317787  11.74164772]
   [ 12.61811256  13.58873653]
   [ 14.42555618  15.38725948]]

  [[ 16.18030167  17.13208389]
   [ 17.8778286   18.81876755]
   [ 19.51431465  20.44356918]
   [ 21.08664513  22.00345612]
   [ 24.02982712  24.91982079]
   [ 25.3978138   26.27359962]
   [ 26.69583511  27.55699158]
   [ 27.92391014  28.7700882 ]]]]
"""
```

### dropout

> 它强迫一个神经单元，和随机挑选出来的其他神经单元共同工作，达到好的效果。消除减弱了神经元节点间的联合适应性，增强了泛化能力。

```python

def testDropOut():
    """
    dropout解释：输入参数中保留keep_pro参数，未保留参数致0，保留参数 输出为 input/keep_pro
    部分输入舍弃，同时“强化”其他input，个人理解：调0就不用解释是干啥了，调大参数的不仅是为了“加快”训练速度	还一定程度上保证了“和”的稳定性，相当于是把周围的值给“拿”过来
    :return:
    """
    array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "float32")
    dropout = tf.nn.dropout(array, keep_prob=0.7)
    with tf.Session() as sess:
        print("--------------------------------------------------------")
        print(sess.run(dropout))

"""
--------------------------------------------------------
[  0.           2.85714293   4.28571415   0.           0.           8.5714283
  10.          11.4285717   12.8571434   14.28571415]
"""
```

### BATCH NORMALIZATION

> 将我们的每层数据进行一个标准化操作


```python
def testBN():
    """
    BN是用来解决因为层数太多出现梯度弥散的问题，因为调整了每层数据的取值，从而使大的不那么大，让所有的数据都能被激活函数给进行
    
    划分（参考：轻轻碰和重重碰），有效地让每个值都参与了训练
    tensorflow.layers.batch_normalization是集成了之前 tf.nn.moments 和tf.nn.batch_normalization两个方法
    tf.nn.moments ： 求平均值函数 ，注意该方法只接受tf.float32数据类型
    tf.nn.batch_normalization :  BN操作
    :return:
    """
    array = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]], dtype="float32")  #
    # array = tf.reshape(array, shape=[1, 1, 10, 1])
    # bn = tf.nn.batch_normalization(array)
    # bn = tf.layers.batch_normalization(array)
    array = tf.cast(array, dtype=tf.float32)
    mean, variance = tf.nn.moments(array, axes=[0])  # 按维度0求均值和方差
    # mean, variance = tf.nn.moments(array, axes=[0,1])  # 按所有数据求均值和方差
    #bn_divide = tf.nn.batch_normalization(array,mean=mean,variance=variance,)
    bn = tf.layers.batch_normalization(array) #因为涉及到参数的初始化（偏移量和微小正数），所以需要有一个init的步骤

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("------------------mean---------------------")
        print(sess.run(mean))
        print("------------------variance-----------------")
        print(sess.run(variance))
        print("------------------bn（分开操作）-------------")
        print(sess.run(bn))
        

"""
------------------mean---------------------
10.5
------------------variance-----------------
33.25
------------------bn（分开操作）-------------
[[-0.64748418 -0.47406477 -0.30064535 -0.127226    0.04619336  0.21961284
   0.39303219  0.56645155  0.73987091  0.91329026]
 [ 1.08670974  1.26012921  1.43354845  1.60696793  1.78038716  1.95380664
   2.12722611  2.30064535  2.47406483  2.64748406]]
------------------bn（api）-------------
[[  0.99950033   1.99900067   2.99850106   3.99800134   4.99750185
    5.99700212   6.9965024    7.99600267   8.99550343   9.9950037 ]
 [ 10.99450397  11.99400425  12.99350452  13.9930048   14.99250507
   15.99200535  16.99150658  17.99100685  18.99050713  19.9900074 ]]
"""
```

