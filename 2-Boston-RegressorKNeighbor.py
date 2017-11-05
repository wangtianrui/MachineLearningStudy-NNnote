from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

boston = load_boston()
print(boston.DESCR)
x = boston.data
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=33, test_size=0.25)

print("the max target value is", np.max(boston.target))
print("the min target value is", np.min(boston.target))
print("the average target value is", np.mean(boston.target))

# 标准化
ss_x = StandardScaler()
ss_y = StandardScaler()

x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)

y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

# 平均回归
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(x_train, y_train)
uni_knr_y_predict = uni_knr.predict(x_test)

# 根据距离进行加权回归训练
dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(x_train, y_train)
dis_knr_y_predict = dis_knr.predict(x_test)

print("----------------------------------------------------------")
print("uni_Accuracy:", uni_knr.score(x_test, y_test))
print("the mean squared error of uni:",
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))
print("the mean absoluate error of uni:",
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))
print("----------------------------------------------------------")
print("dis_Accuracy:", dis_knr.score(x_test, y_test))
print("the mean squared error of dis:",
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))
print("the mean absoluate error of dis:",
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))