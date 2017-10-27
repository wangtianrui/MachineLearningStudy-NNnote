from sklearn.datasets import load_digits  # 手写数字的加载包
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

digits = load_digits()  # 读取数据
print(digits.data.shape)  # (1797, 64) size of 8x8


X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


#标准化数据，保证每个维度的特征数据方差为1，均值为0，使预测结果不会被某些维度过大的特征值而主导
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


#初始化向量机
lsvc = LinearSVC()
#训练
lsvc.fit(X_train,Y_train)
#预测
y_predict=lsvc.predict(X_test)
#使用自带的“打分”
print("Accuracy :",lsvc.score(X_test,Y_test))

print(classification_report(Y_test,y_predict,target_names=digits.target_names.astype(str)))
