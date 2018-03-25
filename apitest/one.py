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


if __name__ == "__main__":
    # testMatmul()
    testSoftMaxAndCross_entropy()
