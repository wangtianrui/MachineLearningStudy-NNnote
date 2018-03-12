import matplotlib.pyplot as plt
from pylab import *

DATA_HOME = "E:/python_programes/MachineLearningStudy/datas/tree/"


def createDataSet():
    """
    创建测试数据集
    :return:
    """
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    """
    获得数据集中某个属性（axis）为value的所有集合
    :param dataSet:
    :param axis:
    :param value:
    :return:
    """
    reDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            reDataSet.append(reducedFeatVec)
    return reDataSet


def calcShannonEnt(dataSet):
    """
    shannon信息熵
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)
    labelCount = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)  # 注意这里取负号的技巧,因为是求和后求负，所以可以直接遍历进行连减、
    return shannonEnt


def chooseBestFeatureToSplit(dataSet):
    """
    选择出信息增益最大化的特征
    :param dataSet:
    :return:
    """
    numFeature = len(dataSet[0]) - 1  # 减去一个label
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeature):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntory = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntory += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntory
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """
    使用分类名称的列表，然后创建键值为classList中唯一值的数据字典，
    字典对象存储了classList中每个类标签出现的频率最后进行排序，返
    回出现次数最多的分类名称
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(),
                                  key=operator.itemgetter(1),
                                  everse=True)  # 降序，按第2列数据排序
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    用迭代,创建树
    :param dataSet:
    :param labels:
    :return:
    """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0] == len(classList)):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]  # 获得所有“最佳特征”的值，如：颜色：白色、红色等
    myTree = {bestFeatLabel: {}}  # 创建一个包含字典的字典
    del (labels[bestFeat])  # 将已经添加的Label给去掉
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    剪头线段的模型
    :param nodeTxt:文本
    :param centerPt:注解框位置
    :param parentPt:起点位置
    :param nodeType:文本框类型
    :return:
    """
    arrow_args = dict(arrowstyle="<-")  # 剪头符号
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, \
                            xycoords='axes fraction', \
                            xytext=centerPt, textcoords='axes fraction', \
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlot():
    """
    绘制
    :return:
    """
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 让中文通过编译
    fig = plt.figure(5, facecolor="white")
    fig.clf()  # Clear the figure.
    createPlot.ax1 = plt.subplot(111, frameon=False)
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # boxstyle = "swatooth"意思是注解框的边缘是波浪线型的，fc控制的注解框内的颜色深度
    leafNode = dict(boxstyle="round4", fc="0.8")  # 平滑的线
    # 定制文本框
    plotNode('决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


def getNumLeafs(myTree):
    """
    得到叶子数
    :param myTree:
    :return:
    """
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    """
    得到树的深度
    :param myTree:
    :return:
    """
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def retrieveTree(i):
    listOfTrees = [
        {
            'no surfacing':
                {
                    0: 'no',
                    1: {
                        'flippers':
                            {
                                0: 'no',
                                1: 'yes'
                            }
                    }
                }
        },
        {
            'no surfacing':
                {
                    0: 'no',
                    1: {
                        'flippers':
                            {
                                0: {
                                    'head':
                                        {
                                            0: 'no',
                                            1: 'yes'
                                        }
                                },
                                1: 'no'
                            }
                    }
                }
        }
    ]

    return listOfTrees[0]


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot().ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.x0ff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.y0ff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # boxstyle = "swatooth"意思是注解框的边缘是波浪线型的，fc控制的注解框内的颜色深度
    leafNode = dict(boxstyle="round4", fc="0.8")  # 平滑的线
    plotNode(firstStr, cntrPt, parentPt, decisionNode)


if __name__ == "__main__":
    # testMat, testLabel = createDataSet()
    # print(calcShannonEnt(testMat))  # 0.9709505944546686
    # print(chooseBestFeatureToSplit(testMat))
    # createPlot()
    listOfTree = retrieveTree(1)
    numLeaf = getNumLeafs(listOfTree)
    depth = getTreeDepth(listOfTree)
    print(listOfTree, numLeaf, depth)
