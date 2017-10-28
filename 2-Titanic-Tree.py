import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
print(titanic.head())

X = titanic[['pclass', 'age', 'sex']]
Y = titanic['survived']

print(X.info())

X['age'].fillna(X['age'].mean(), inplace=True)
print(X.info())

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=33)
vec = DictVectorizer(sparse=False)

x_train = vec.fit_transform(x_train.to_dict(orient='record'))
print(vec.feature_names_)

x_test = vec.transform(x_test.to_dict(orient='record'))
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_predict=dtc.predict(x_test)
print("Accuracy:",dtc.score(x_test,y_test))
print(classification_report(y_test,y_predict,target_names=['died','survived']))