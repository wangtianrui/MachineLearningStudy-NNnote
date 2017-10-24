import pandas as pd
import numpy as np
from sklearn.cross_validation import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

column_names=['Sample code number','Clump Thickness','Uniformity of Cell Size'
    ,'Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size'
    ,'Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
                   ,names=column_names)


#NaN, Not a Number, 非数. 它即不是无穷大, 也不是无穷小, 而是numpy觉得无法计算时返回的一个符号
#无穷大减无穷大会导致NaN
#无穷大乘以0或无穷小或除以无穷大会导致NaN
#有NaN参与的运算, 其结果也一定是NaN
data = data.replace(to_replace='?',value=np.nan)
#Replace values given in 'to_replace' with 'value'.


#丢掉有缺失值的数据
data = data.dropna(how='any')

print(data.shape)

#用于分割数据集
# train_test_split 将数组或矩阵拆分为随机列和测试子集
'''
random_state:编号，随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。
比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：
种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。

比如：random_state=非0数   那么只要每次数字一样就能得到同样的分类
      random_state=0   那么每次都是不同的
'''
X_train , X_test  , y_train , y_test = train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25
                                                        ,random_state=0)

print(y_train.value_counts())
print(y_test.value_counts())


#标准化数据，保证每个维度的特征数据方差为1，均值为0，使预测结果不会被某些维度过大的特征值而主导
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


#LogisticRegression中的fit函数/模块用来训练模型参数
lr=LogisticRegression()
#梯度下降
sgdc=SGDClassifier()

lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)

sgdc.fit(X_train,y_train)
sgdc_y_predict=sgdc.predict(X_test)

#一般的医学预测不能只用误差值来进行train，因为模型判断出来的结果和label有2x2的组合，其中最不希望看到的是“有病，判出没病”
#所以需要引入另外的评价指标（召回率）

print('Accuracy :',lr.score(X_test,y_test))
print(classification_report(y_test,lr_y_predict,target_names=['Benign','Malignant']))

print('Accuracy :',sgdc.score(X_test,y_test))
print(classification_report(y_test,sgdc_y_predict,target_names=['Benign','Malignant']))
