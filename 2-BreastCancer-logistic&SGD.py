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

X_train , X_test  , y_train , y_test = train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25
                                                        ,random_state=33)

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
