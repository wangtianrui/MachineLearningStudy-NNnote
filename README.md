# 三本书的学习：《kaggle实战》(book1)、《机器学习实战》(ML)、基于python的数据结构，外加2018暑假学习和一些比赛总结

## 一些常见库的使用:

* #### python自带api：

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

  > 文字拆分

  ```python
  string(一个String对象).split('.')[0]
  #将string按.进行拆分，取第一部分
  ```

  > 矩阵选取

  ```python
  trainMat[i,:]   #选取第一行
  ```

  > 排序(sort)

  ```python
   sorted(iterable, cmp=None, key=None, reverse=False) 
      """
      iterable：是可迭代类型;
  	cmp：用于比较的函数，比较什么由key决定;
  	key：用列表元素的某个属性或函数进行作为关键字，有默认值，迭代集合中的一项;
  	reverse：排序规则. reverse = True  降序 或者 reverse = False 升序，有默认值。
  	返回值：是一个经过排序的可迭代类型，与iterable一样。
      """
  ```

  > 遍历获得List(List操作)

  ```python
  classList = [example[-1] for example in dataSet]  
  #用example遍历dataSet，然后取出每个example的最后一个数据放入classList
  ```

  > 字典内部包含字典(可以用于创建树)

  ```py
   myTree = {bestFeatLabel: {}}  # 创建一个包含字典的字典
  ```

  > 删除变量(del )

  ```python
      a=1       # 对象 1 被 变量a引用，对象1的引用计数器为1  
      b=a       # 对象1 被变量b引用，对象1的引用计数器加1  
      c=a       #1对象1 被变量c引用，对象1的引用计数器加1  
      del a     #删除变量a，解除a对1的引用  
      del b     #删除变量b，解除b对1的引用  
      print(c)  #最终变量c仍然引用1  
      
      """
      a=1       # 对象 1 被 变量a引用，对象1的引用计数器为1  
      b=a       # 对象1 被变量b引用，对象1的引用计数器加1  
      c=a       #1对象1 被变量c引用，对象1的引用计数器加1  
      del a     #删除变量a，解除a对1的引用  
      del b     #删除变量b，解除b对1的引用  
      print(c)  #最终变量c仍然引用1  
      """
  ```

  >对象的类型

  ```python
  if type(secondDict[key]).name=='dict'  #测试该对象是不是字典
  ```

  > 循环中对item操作

  ```python
  a = [[1,2],[2,3],[4,5]]
  b = [item[1] for item in a]
  print(b)
  # [ 2,3,5 ]
  ```

  ​

  ​

  ​

  ​					


* #### operator (import operator):

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


> 获取对象单维数据(itemgetter)

```python
a = [1,2,3] 
b=operator.itemgetter(1)      //定义函数b，获取对象的第1个域的值
#2

b=operator.itemgetter(1,0)  //定义函数b，获取对象的第1个域和第0个的值
b(a) 
#(2, 1)

要注意，op
```



* ##### numpy (import numpy as np)

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

  > 矩阵除法

  ```python
   np.linalg.solve(a,b)
  """
  solve函数有两个参数a和b。a是一个N*N的二维数组，而b是一个长度为N的一维数组，solve函数找到一个长度为N的一维数组x，使得a和x的矩阵乘积正好等于b，数组x就是多元一次方程组的解
  """
  ```

  * #### matplotlib ：

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

  * annotations

  * #### 文件操作封装类 : os

    ```txt
    Python的标准库中的os模块包含普遍的操作系统功能。如果你希望你的程序能够与平台无关的话，这个模块是尤为重要的。即它允许一个程序在编写后不需要任何改动，也不会发生任何问题，就可以在Linux和Windows下运行。

    下面列出了一些在os模块中比较有用的部分。它们中的大多数都简单明了。

    os.sep可以取代操作系统特定的路径分隔符。windows下为 “\\”
    os.name字符串指示你正在使用的平台。比如对于Windows，它是'nt'，而对于Linux/Unix用户，它是'posix'。

    os.getcwd()函数得到当前工作目录，即当前Python脚本工作的目录路径。

    os.getenv()获取一个环境变量，如果没有返回none

    os.putenv(key, value)设置一个环境变量值

    os.listdir(path)返回指定目录下的所有文件和目录名。

    os.remove(path)函数用来删除一个文件。

    os.system(command)函数用来运行shell命令。

    os.linesep字符串给出当前平台使用的行终止符。例如，Windows使用'\r\n'，Linux使用'\n'而Mac使用'\r'。

    os.curdir:返回当前目录（'.')
    os.chdir(dirname):改变工作目录到dirname

    ========================================================================================

    os.path常用方法：

    os.path.isfile()和os.path.isdir()函数分别检验给出的路径是一个文件还是目录。

    os.path.existe()函数用来检验给出的路径是否真地存在

    os.path.getsize(name):获得文件大小，如果name是目录返回0L

    os.path.abspath(name):获得绝对路径
    os.path.normpath(path):规范path字符串形式

    os.path.split(path) ：将path分割成目录和文件名二元组返回。

    os.path.splitext():分离文件名与扩展名

    os.path.join(path,name):连接目录与文件名或目录;使用“\”连接
    os.path.basename(path):返回文件名
    os.path.dirname(path):返回文件路径
    ```

    ​

    > 路径拼接

    ```python 
    os.path.join
    """
    函数功能：连接两个或更多的路径名组件

    如果各组件名首字母不包含'/'，则函数会自动加上

    如果有一个组件是一个绝对路径，则在它之前的所有组件均会被舍弃

    如果最后一个组件为空，则生成的路径以一个'/'分隔符结尾
    """
    ```

    > 获取目录内容

    ```python
    os.listdir(filename)#获得子文件名
    #Return a list containing the names of the files in the directory.
    ```

    ```python
    import os

    print '***获取当前目录***'
    print os.getcwd()
    print os.path.abspath(os.path.dirname(__file__))

    print '***获取上级目录***'
    print os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    print os.path.abspath(os.path.dirname(os.getcwd()))
    print os.path.abspath(os.path.join(os.getcwd(), ".."))

    print '***获取上上级目录***'
    print os.path.abspath(os.path.join(os.getcwd(), "../.."))
    ```

* ##### threading(多线程)


  ```python
  t1 = threading.Thread(target=run_thread, args=(5,))  #target为方法名，args为参数的元组表达形式
  t2 = threading.Thread(target=run_thread, args=(8,))
  t1.start()
  t2.start()
  t1.join()
  t2.join(
  ```

  ​

  
