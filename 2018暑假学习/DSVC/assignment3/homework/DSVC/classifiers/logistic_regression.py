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

    def loss(self, X_batch, y_batch):
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
            # 梯度下降的loss
            # print(y_batch[i], "----", np.dot(self.w.T, X_batch[i]))
            #print(np.dot(self.w.T, X_batch[i]))#NAN值的是他
            loss += -(y_batch[i] * (np.dot(self.w.T, X_batch[i]))) + np.log(1 + np.exp(np.dot(self.w.T, X_batch[i])))
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        # size_w = X_batch.shape[0]
        # size_h = X_batch.shape[1]
        gradients = 0
        dot = np.dot(X_batch, self.w)
        logists = sigmod(dot)
        diff = y_batch - logists
        # print(y_batch[0],"----",logists[0])
        # for i in range(self.w.shape[0]):
        #    weights_gradients.append(np.mean(grands)) #一个batch取均值
        # for index in range(X_batch.shape[0]):
        #     for i in range(self.w.shape[0]):
        #         list(weights_gradients[index]).append(np.dot(X_batch[index].T, diff))
        #         weights_gradients[index][-1] = diff
        # weights_gradients = np.sum(weights_gradients, axis=0)
        for index in range(X_batch.shape[0]):
            gradients += np.dot(X_batch[index], diff[index])
        return loss, gradients/X_batch.shape[0]  #取均值免得步长过大直接nan
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
        # random.seed(100)
        # np.random.shuffle(all_data)

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
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient

            loss, grad = self.loss(np.array(X_batch), np.array(y_batch))
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

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

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
        slicer = 0.5 #设置阈值
        y_pred = sigmod(np.dot(X,self.w))+slicer #因为astype是直接舍不会入，所以加上阈值
        y_pred = y_pred.astype(int)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def one_vs_all(self, X, y, learning_rate=1e-3, num_iters=100,
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

        """
