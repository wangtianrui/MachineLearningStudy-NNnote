import numpy as np


def lwlr(testPoint, xArr, yArr, k=1.0):
    X = np.array(xArr)
    y = np.array(yArr)
    len = len(X)
    weights = np.eye(len)
    for j in range(len):
        diff = testPoint - X[j, :]  # diff = [ x1-x2 , y1-y2 ]
        weights[j, j] = np.exp(np.dot(diff, diff.T) / (-2.0 * k ** 2))


if __name__ == "__main__":
    lwlr()
