# coding=utf-8
import numpy as np
import random
import math


def sigmod(X):
    """
    sigmod
    :return: 返回输入值对应的sigmod结果
    """
    #     sigmod_results = []
    #     for i in range(X.shape[0]):
    #         try:
    #             sigmod_results.append(1.0 / (1 + math.exp(-X[i])))
    #         except OverflowError :
    #             sigmod_results.append(0)
    return 1.0 / (1 + np.exp(-X))


class LogisticRegression(object):

    def __init__(self):
        self.w = None
        self.ws = None

    def loss(self, X_batch, y_batch, learning_rate=1e-3, one_vs_all_index=-1, reg=True):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        #########################################################################
        # TODO:                                                                 #
        # calculate the loss and the derivative                                 #
        #########################################################################
        loss = 0
        for i in range(X_batch.shape[0]):
            if one_vs_all_index == -1:
                loss += -(y_batch[i] * (np.dot(self.w.T, X_batch[i]))) + np.log(
                    1 + np.exp(np.dot(self.w.T, X_batch[i])))
            else:
                if reg:
                    reg = (learning_rate / 2 * X_batch.shape[0]) * np.sum(np.power(self.ws[one_vs_all_index], 2))
                    loss += -(y_batch[i] * (np.dot(self.ws[one_vs_all_index].T, X_batch[i]))) + np.log(
                        1 + np.exp(np.dot(self.ws[one_vs_all_index].T, X_batch[i]))) + reg
                else:
                    loss += -(y_batch[i] * (np.dot(self.ws[one_vs_all_index].T, X_batch[i]))) + np.log(
                        1 + np.exp(np.dot(self.ws[one_vs_all_index].T, X_batch[i])))
        gradients = np.zeros(785)
        if one_vs_all_index == -1:
            dot = np.dot(X_batch, self.w)
        else:
            dot = np.dot(X_batch, self.ws[one_vs_all_index])
        logists = sigmod(dot)
        diff = y_batch - logists
        for index in range(X_batch.shape[0]):
            if one_vs_all_index != -1:
                if reg:
                    dot = np.dot(X_batch[index], diff[index])
                    gradients[1:] += dot[1:] + (learning_rate / X_batch.shape[0]) * self.ws[one_vs_all_index][1:]
                    gradients[0] += dot[0]
                else:
                    gradients += np.dot(X_batch[index], diff[index])
            else:
                gradients += np.dot(X_batch[index], diff[index])

        return loss, gradients / X_batch.shape[0]  # 取均值免得步长过大直接nan
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def train(self, X, y, learning_rate=1e-3, num_iters=100,
              batch_size=200, verbose=True):

        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape

        if self.w is None:
            self.w = 0.001 * np.random.randn(dim)

        loss_history = []

        # 将feature与label连起来，方便后面batch的划分
        all_data = list(zip(X, y))

        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            # batch_data = np.random.choice(all_data, batch_size, False) 
            # error: ValueError: a must be 1-dimensional 
            # 查询相关api貌似该方法不能用于数组中元素为元组情况下的选取
            batch_data = random.sample(all_data, batch_size)
            X_batch, y_batch = zip(*batch_data)
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient

            loss, grad = self.loss(X_batch, y_batch)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            self.w += float(learning_rate) * np.array(grad)
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and (it % 1000 == 0 or it == num_iters - 1):
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X, slicer = 0.5, one_vs_all=False):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.
        - slicer : threshold value
        - one_vs_all : a flag to differentiate one_vs_all and two-category
        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is an integer giving the predicted
        class.
        """
        y_pred = np.zeros(X.shape[1])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        # slicer = 0.7  # 设置阈值，阈值为1-sclicer,经过测试0.7效果最好，对应的阈值是0.3
        # 这里还发现一个小技巧，调整阈值可以根据loss来进行调整，如果最终loss偏大，则阈值对应调小
        if one_vs_all == False:
            y_pred = sigmod(np.dot(X, self.w)) + slicer  # 因为astype是直接舍不会入，所以加上slicer值然后调用astype就相当于使用（1-slicer)作为阈值处理
            y_pred = y_pred.astype(int)
        else:
            y_pred = sigmod(np.dot(X, self.ws.T)) + slicer  # 因为astype是直接舍不会入，所以加上阈值
            y_pred = y_pred.astype(int)
            y_pred = np.argmax(y_pred, axis=1)  # 每行取出最大值的下标

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def one_vs_all(self, X, y, learning_rate=1e-3, num_iters=100,
                   batch_size=200, verbose=True, reg=True):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        """
        dim = X.shape[1]
        if self.ws == None:
            self.ws = 0.001 * np.random.randn(10, dim)  # 以正态分布形式初始化10个分类器的参数

        all_loss_history = []
        datas = list(zip(X, y))

        for i in range(num_iters):
            data_batch = random.sample(datas, batch_size)
            X_batch, y_batch = zip(*data_batch)
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            for index in range(10):
                label = (index == y_batch).astype(int)  # 将非index的标签转换为0
                # print(label)
                loss, grad = self.loss(X_batch, label, learning_rate=learning_rate, one_vs_all_index=index, reg=reg)
                self.ws[index] += float(learning_rate) * grad
                try:
                    all_loss_history[index].append(loss)
                except Exception:
                    all_loss_history.append([loss])
            if verbose and (i % 1000 == 0 or i == num_iters - 1):
                loss_mean = np.mean([x[-1] for x in all_loss_history])
                # 这里发现 平均Loss能很快降到10以下，但是到5左右就基本是来回变了，所以8以下后降低学习率
                # 此方法将准确率从0.814提升到了0.846
                if loss_mean < 8:
                    learning_rate = 1e-7
                print('iteration %d / %d ' % (i, num_iters), " loss_array_mean", loss_mean)
        return all_loss_history
