import numpy as np


def loadDataSet():
    """
    生成数据集以及标签
    :return:
    """
    postingList = [['my', 'dog', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'T', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1是侮辱性，0是正常
    return postingList, classVec


def createVocabList(dataSet):
    """
    通过数据集生成一个无重复的词条Set
    :param dataSet:
    :return:
    """
    vocabSet = set([])  # 构造一个空的集合
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 集合并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    查看输入文本中是否出现词条表对应的单词
    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0] * len(vocabList)
    # print("returnVec", returnVec)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word : %s is not in my Vocabulary!" % word)
    return returnVec


def trainNBO(trainMatrix, trainCategory):
    """
    训练模型
    :param trainMatrix: 二维矩阵
    :param trainCategory: labels
    :return:
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num + - trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom  # 对每个做除法
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive


if __name__ == "__main__":
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    # print(myVocabList)
    # print(setOfWords2Vec(myVocabList, listOPosts[3]))
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    print(trainMat)
    p0V, p1V, pAb = trainNBO(trainMat, listClasses)
    print(p0V)
    print(p1V)
    print(pAb)
