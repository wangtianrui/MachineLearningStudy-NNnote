import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']

# 如果缺失年龄信息，我们就用年龄的平均值来填补，保证训练完整并不影响
x['age'].fillna(x['age'].mean(), inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

vec = DictVectorizer(sparse=False)

# 通过指数映射和转化为特征向量
# Learn a list of feature name -> indices mappings and transform X.
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
# 转换为稀疏矩阵
x_test = vec.fit_transform(x_test.to_dict(orient='record'))

# 使用单树分类器
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
y_predict_tree = dtc.predict(x_test)

# 使用随机法森林分类器
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_predict_rfc = rfc.predict(x_test)

# 梯度法森林
gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
y_predict_gbc = gbc.predict(x_test)

print("Accuracy:",dtc.score(x_test,y_test))
print(classification_report(y_test,y_predict_tree))

print("Accuracy:",rfc.score(x_test,y_test))
print(classification_report(y_test,y_predict_rfc))

print("Accuracy:",gbc.score(x_test,y_test))
print(classification_report(y_test,y_predict_gbc))
