import tensorflow as tf
import numpy as np


def testMatmul():
    """
    测试 * 运算符和 tf.matmul()的区别
    :return:
    """
    array2x3 = np.array([[1, 2, 3],
                         [4, 5, 6]], dtype=float)

    array3x4 = np.array([[7, 8, 9, 10],
                         [11, 12, 13, 14],
                         [15, 16, 17, 18]], dtype=float)
    print(array2x3)
    print(array3x4)
    # print(array2x3 * array3x4)     error
    array1x5 = np.array([1, 2, 3, 4, 5])
    array1x52 = np.array([6, 7, 8, 9, 10])
    print(array1x5 * array1x52)

    with tf.Session() as sess:
        print(sess.run(tf.matmul(array2x3, array3x4)))


def testSoftMaxAndCross_entropy():
    """
    测试softmax和cross_entropy分开写  和 使用混合的api的区别    ， 测试结果：没区别（batch_size = 1）
    顺便测试了 label 取int和float的区别：无（batch_size = 1）,
    :return:
    """
    # label = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype="float")  # label为8
    label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype="int64")  # label为1
    # array = np.array([0, 1.1, 2.5, 3.34, 4.346436, 5.434, 6.46, 7.2352, 9.234, 10.234], dtype="float")
    array = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
                      [1, 0, 3, 5, 7, 1, 2, 4, 6, 12]], dtype="float")
    mysoftmax = tf.nn.softmax(array)
    mycross = -tf.reduce_sum(label * tf.log(mysoftmax))
    apicross = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=array, labels=label))
    apicrossv2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=array, labels=label))
    with tf.Session() as sess:
        print(sess.run(mysoftmax))
        print("------------------------------------------------------------")
        print(sess.run(mycross))
        print("------------------------------------------------------------")  # label int 1.36922634586
        print(sess.run(apicross))
        print("------------------------------------------------------------")
        print(sess.run(apicrossv2))


def testLogSotfMax():
    array = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
                      [1, 0, 3, 5, 7, 1, 2, 4, 6, 12]], dtype="float")
    log_softmax = tf.nn.log_softmax(array)
    my_softmax = tf.nn.softmax(array)
    with tf.Session() as sess:
        print(sess.run(log_softmax))
        print("------------------------------------------------------------")
        print(sess.run(my_softmax))


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
        print("-------------------------------beta0.05,bias0,alpha0.05-----------------------------")
        print(sess.run(lrn_result2))
        print("-------------------------------beta0.5,bias1,alpha1-----------------------------")
        print(sess.run(lrn_result3))
        print("------------------------------good parameter------------------------------")
        print(sess.run(lrn_good))

if __name__ == "__main__":
    # testMatmul()
    # testSoftMaxAndCross_entropy()
    # testLogSotfMax()
    testLRN()
