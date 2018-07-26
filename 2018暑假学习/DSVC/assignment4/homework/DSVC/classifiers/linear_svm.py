# coding=utf-8
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    # print(y)
    # compute the loss and the gradient
    num_classes = W.shape[0]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W.T)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin

                dW[j] += X[j]
                dW[y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # l2w = np.sqrt(np.sum(np.square(W)))
    # labels = []
    # for i in range(num_classes):
    #     labels.append(np.array((i == y)).astype(int))
    # labels = np.array(labels)
    # labels[np.where(labels == 0)] = -1
    # labels = labels.T
    # y_pred = np.dot(X, W.T)  # 500x10
    # # print(y_pred.shape)
    # # print((y_pred <= 1).astype(int))
    # # print( * labels[i] * X)
    # # print(labels.shape)
    # # print(y_pred.shape)
    #
    # # for i in range(num_classes):
    # #     # dW[:, i] = np.mean((y_pred <= 1).astype(int).T[:, i] * labels[:, i] * X[:i], axis=1).T + W[:, i]
    # #     dW[i] = dW[i] + (labels[:,i] * y_pred <= 1).astype(int) * labels[i]
    # # dW = np.dot((((labels * y_pred) <= 1).astype(int) * labels).T, X) / num_train + W
    # delta = 0
    # for i in range(num_train):
    #     for j in range(num_classes):
    #         delta += (int(y_pred[i][y[i]] + y_pred[i][j]) <= 1) * X[i]
    # dW = delta / num_train + W
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
