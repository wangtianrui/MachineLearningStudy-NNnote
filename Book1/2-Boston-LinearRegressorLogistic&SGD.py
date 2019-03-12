from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
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

y_train = ss_y.fit_transform(y_train.reshape(-1,1))
y_test = ss_y.transform(y_test.reshape(-1,1))

lr = LinearRegression()
lr.fit(x_train, y_train)
lr_y_predict = lr.predict(x_test)

sgdr = SGDRegressor()
sgdr.fit(x_train, y_train)
sgdr_y_predict = sgdr.predict(x_test)


print("R-squared:",r2_score(y_test,lr_y_predict))
print("mean-squared-error:",mean_squared_error(y_test,lr_y_predict))
print("mean-absoluate-error:",mean_absolute_error(y_test,lr_y_predict))


print("R-squared:",r2_score(y_test,sgdr_y_predict))
print("mean-squared-error:",mean_squared_error(y_test,sgdr_y_predict))
print("mean-absoluate-error:",mean_absolute_error(y_test,sgdr_y_predict))