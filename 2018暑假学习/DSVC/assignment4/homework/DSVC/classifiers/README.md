### 利用梯度优化SVM

$$
f(ω)=\min \frac{\lambda}{2}||\omega||^2+\frac {1}{m}∑l(ω,(x,y))
$$

$$
l(\omega,(x,y))=\max{(0,1-y<w,x>)}
$$



使用随机梯度下降解目标函数的算法称作pegasos。在每一次的迭代中，随机挑选一个训练样本计算目标函数的梯度，然后在在相反的方向走一个预订好的步长。

* 随机选取一个训练样本it其中i代表选取的样本,t代表迭代的带入$f(\omega)$
* 得到近似的目标函数

$$
f(\omega,it)=\frac{\lambda}{2}||\omega||^2+l(\omega,(x_{it},y_{it}))
$$

* 计算梯度得

$$
\frac {\partial J}{\partial w} = -\frac{1}{m}\sum_{i=1}^m \amalg(y_{i}(\omega^Tx_{i}+b)\leq1) \cdot y_{i}x_{i} + \lambda\omega
$$

$$
\frac {\partial J}{\partial b} = -\frac{1}{m}\sum_{i=1}^m \amalg(y_{i}(\omega^Tx_{i}+b)\leq1) \cdot y_{i}
$$

$$
w_{new} = w - \eta \frac{\partial J}{\partial w}
$$

$$
b_{new}=b-\eta\frac{\partial J}{\partial b}
$$

