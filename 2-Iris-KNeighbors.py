from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

iris = load_iris()
print(iris.data.shape)
print(iris.DESCR)

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)
ss = StandardScaler()

x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

knc = KNeighborsClassifier()
knc.fit(x_train, y_train)
y_predict = knc.predict(x_test)
print("Accuracy:",knc.score(x_test,y_test))
print(classification_report(y_test,y_predict,target_names=iris.target_names))
