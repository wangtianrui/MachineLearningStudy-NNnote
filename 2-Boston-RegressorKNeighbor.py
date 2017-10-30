from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
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



#平均回归
uni_knr = KNeighborsRegressor(weights='uniform')
