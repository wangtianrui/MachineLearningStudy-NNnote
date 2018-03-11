import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
import os

DATA_HOME = "E:/python_programes/MachineLearningStudy/datas/knn/"


def createDataset():
    """
    :return: 生成样本
    """
    group = np.array(
        [
            [1.0, 1.1],
            [1.0, 1.0],
            [0, 0],
            [0, 0.1]
        ]
    )
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    '''
    :param inX: 输入（待分类的样本）
    :param dataSet: 已分类的数据集
    :param labels: 已分类的数据集对应的标签
    :param k: k值
    :return: 待分类的样本的预测类别
    '''
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 因为要与四个点计算。所以在纵向上x4,[(x1-xi),(y1-yi)]
    sqDiffMat = diffMat ** 2  # [(x1-xi)^2,(y1-yi)^2],^为平方
    sqDistances = sqDiffMat.sum(axis=1)  # 横向求和 ,axis为参考轴, axis=1则按照纵向为方向进行挨着遍历(从上往下，横向求和)
    distance = sqDistances ** 0.5
    sortedDistIndicies = distance.argsort()  # 排序，返回矩阵中对应的index，方便后面直接对对应的label进行提取
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 得到每个label
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 在字典中给对应的label+1
        print(classCount)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 按数量排序
    return sortedClassCount[0][0]


def file2matrix(filename):
    """

    :param filename:
    :return:
    """

    fr = open(filename)
    arrayOLines = fr.readlines()  # 返回一个列表，包括所有行
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 截取掉所有回车
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))  # -1可以索引最后一列数据
        index += 1
    return returnMat, classLabelVector


def draw(datingDataMat, datingLabels):
    """
    绘制图像
    :param datingDataMat:
    :return:
    """
    fig = plt.figure()  # 绘图对象
    ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])  # 所有的1index，和2index数据加载
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * np.array(datingLabels),
               15.0 * np.array(datingLabels))  # 个性化标点
    plt.show()


def autoNorm(dataset):
    minVals = dataset.min(0)
    maxVals = dataset.max(0)
    ranges = maxVals - minVals
    normDataset = np.zeros(dataset.shape)
    m = dataset.shape[0]
    print(m)
    normDataSet = dataset - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # 注意：操作应该要有一种“集合”整体操作的思想，而不是单体
    # normDataSet = np.linalg.solve(normDataSet, np.tile(ranges, (m, 1)))  # 矩阵除法
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.1  # k值（0.1为百分之10）
    # filename = "E:/python_programes/MachineLearningStudy/datas/knn/datingTestSet2.txt"
    filename = os.path.join(DATA_HOME, "datingTestSet2.txt")
    datingDataMat, datingLabels = file2matrix(filename)
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    m = datingDataMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(datingDataMat[i, :], datingDataMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        print("result is %d , label is %d" % (classifierResult, datingLabels[i]))

        if classifierResult != datingLabels[i]:
            errorCount += 1
    print("errors:", errorCount)


def img2vector(filename):
    """
    将32x32的“图”转换成(1,1024)矩阵
    :param filename:
    :return:
    """
    filename = os.path.join(DATA_HOME, filename)
    returnVector = np.zeros(shape=(1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0, 32 * i + j] = int(lineStr[j])
    return returnVector


def handwritingClassTest():
    """
    手写数字训练测试
    :return:
    """
    hwLabels = []
    trainFileList = os.listdir(os.path.join(DATA_HOME, "trainingDigits"))
    print(trainFileList[1])
    m = len(trainFileList)
    trainMat = np.zeros(shape=(m, 1024))
    for i in range(m):
        fileNameStr = trainFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainMat[i, :] = img2vector(os.path.join(DATA_HOME, "trainingDigits", fileNameStr))
    testFileList = os.listdir(os.path.join(DATA_HOME, "testDigits"))
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        vectorUnderTest = img2vector(os.path.join(DATA_HOME, "testDigits", fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainMat, hwLabels, 3)
        print("result:", classifierResult, "label:", classNumStr)
        print("error:", errorCount)


if __name__ == '__main__':
    """
    group, labels = createDataset()
    result = classify0([0, 0], group, labels, 3)
    print(result)
    
    datingClassTest()
    """
    handwritingClassTest()
