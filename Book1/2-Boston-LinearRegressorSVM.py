from sklearn.svm import SVR
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf


boston = load_boston()
print("test:",boston.DESCR)
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

print("------------------------------------",x_train)

linear_svr = SVR(kernel='linear')
linear_svr.fit(x_train, y_train)
linear_svr_predict = linear_svr.predict(x_test)

poly_svr = SVR(kernel='poly')
poly_svr.fit(x_train, y_train)
poly_svr_predict = poly_svr.predict(x_test)

rbf_svr = SVR(kernel="rbf")
rbf_svr.fit(x_train, y_train)
rbf_svr_predict = rbf_svr.predict(x_test)

print("############################################");
print("R-squared:",r2_score(y_test,linear_svr_predict))
print("mean-squared-error:",mean_squared_error(y_test,linear_svr_predict))
print("mean-absoluate-error:",mean_absolute_error(y_test,linear_svr_predict))
print("############################################");
print("R-squared:",r2_score(y_test,poly_svr_predict))
print("mean-squared-error:",mean_squared_error(y_test,poly_svr_predict))
print("mean-absoluate-error:",mean_absolute_error(y_test,poly_svr_predict))
print("############################################");
print("R-squared:",r2_score(y_test,rbf_svr_predict))
print("mean-squared-error:",mean_squared_error(y_test,rbf_svr_predict))
print("mean-absoluate-error:",mean_absolute_error(y_test,rbf_svr_predict))
print("############################################");