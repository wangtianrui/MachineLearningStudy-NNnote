# 两本书的学习：《kaggle实战》(book1)、《机器学习实战》(ML)

## 一些常见库的使用:

* python自带api：

  > 对矩阵进行排序，返回index 

  ```python
  import numpy as np
  x=np.array([1,4,3,-1,6,9])
  x.argsort()

  #输出为y=array([3,0,2,1,4,5])。
  #argsort()函数是将x中的元素从小到大排列，相当于是将以前矩阵中的index进行排序（按照index对应的值），然后输出到y。例如：x[3]=-1最小，所以y[0]=3,x[5]=9最大，所以y[5]=5。
  ```

  > readlines() , readline()

  ```python
  fr = open(filename)
  arrayOLines = fr.readlines()#返回一个列表，包括所有行
  #readline()返回一行
  ```

  > len()

  ```python
  numberOfLines = len(arrayOLines) #测长度
  ```

  ​


* operator (import operator):

  多为基本运算操作:

  > 排序

  ```python 
  sorted()
  #def sorted(*args, **kwargs): Return a new list containing all items from the iterable in ascending order.

  sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
  #第一个参数是一个迭代器（字典的）,第二个则是排序标准
  ```

  ​

  https://www.cnblogs.com/nju2014/p/5568139.html

* numpy (import numpy as np)

  科学计算包，包含了大量的矩阵运算，所有操作都基于numpy数组

  > 创建矩阵

  ```python 
   group = np.array([1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1])
      #def array(p_object, dtype=None, copy=True, order='K', subok=False, ndmin=0):
  ```

  > 复制矩阵进行扩充，重复A  Reps次

  ```python
  np.tile(inX, (dataSetSize, 1))   #def tile(A, reps):
  """
  >>> numpy.tile([0,0],5)#横向重复[0,0]5次，默认纵向1次  
  array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  
  >>> numpy.tile([0,0],(1,1))#横向重复[0,0]1次，纵向1次  
  array([[0, 0]])  
  >>> numpy.tile([0,0],(2,1))#横向重复[0,0]1次，纵向2次  
  array([[0, 0],  
         [0, 0]])  
  >>> numpy.tile([0,0],(3,1))  
  array([[0, 0],  
         [0, 0],  
         [0, 0]])  
  >>> numpy.tile([0,0],(1,3))#横向重复[0,0]3次，纵向1次  
  array([[0, 0, 0, 0, 0, 0]])  
  >>> numpy.tile([0,0],(2,3))#横向重复[0,0]3次，纵向2次
  array([[0, 0, 0, 0, 0, 0],  
         [0, 0, 0, 0, 0, 0]]) 
  """
  ```

  > 生成0矩阵

  ```python
  def zeros(shape, dtype=None, order='C') 
  ```

  * matplotlib ：

    可以用于绘图

    ```python
     fig = plt.figure() #绘图对象
     ax = fig.add_subplot(111)  #创建一个画布，将画布分为1行1列，图像放在第1部分上
     ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])#将
     plt.show()
    """
      def scatter(self, x, y, s=None, c=None, marker=None, cmap=None, norm=None,
                    vmin=None, vmax=None, alpha=None, linewidths=None,
                    verts=None, edgecolors=None,
                    **kwargs):
       将x , y 对应着画到画布上
    """
    ```

    http://blog.csdn.net/anneqiqi/article/details/64125186

