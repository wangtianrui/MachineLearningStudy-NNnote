# Algorithms In Python

## chapter2----python的类

* #### 基本用法

```python
class Person:
    """
    区分一个点：类中变量和对象的变量
    """
    person_number = 0
    
    def __init__(self, strname, strsex, tuplebirthday, strid):
        self._name = strname
        self._sex = strsex
        self._birthday = tuplebirthday
        self._id = strid

    def getName(self):
        return self._name

    def getSex(self):
        return self._sex

    def getBirthday(self):
        return self._birthday

    def getId(self):
        return self._id
    
    def showtest(self):
        print("im super")

    """
    这样写不正确，类内方法必须有self形参
    def show():
        print("show")
    """
    
    """
    如果需要一个函数不带self，也是说这个函数用不着当前对象（不仅仅属于当前这一个对象）就使用staticmethod
    @staticmethod
    def show():
        print("show")
    调用该方法时需要用类名
    Person.show()可以在任意地方调用
    """
    
person = Person("name","sex",(2018,11,1),"1056")
birthday = person.getBirthday()
print(Person.person_number)
```

```text
python中类方法中都会有self形参，因为在对象调用方法的时候会自动将自己作为第一个参数传进去：
person.show() => person.show(person)
如果需要一个函数不带self，也是说这个函数用不着当前对象（不仅仅属于当前这一个对象）就使用staticmethod
```

```text
类中变量和对象变量：
类中的变量需要通过类名访问（如Person中的person_count）：不是当前对象所有，而是类所有
对象变量在类中需要通过self进行访问。在外部通过对象访问
```

* #### 继承

```python
from algorithmsinpython.chapter2.Person import Person

class Student(Person):
    def __init__(self, strname, strsex, tuplebirthday, strdepartment, strid):
        Person.__init__(self, strname, strsex, tuplebirthday, strid)
        self._department = strdepartment

    def getDepartment(self):
        return self._department

    def show(self):
        print(self.getName(), "  ", self.getSex(), "   ", self.getId(), self.getBirthday(), "  ", 				self.getDepartment())



```

* #### super

```python
from algorithmsinpython.chapter2.Person import Person

class Student(Person):
    def __init__(self, strname, strsex, tuplebirthday, strdepartment, strid):
        Person.__init__(self, strname, strsex, tuplebirthday, strid)
        self._department = strdepartment

    def getDepartment(self):
        return self._department

    def show(self):
        super().showtest()
        print(self.getName(), "  ", self.getSex(), "   ", self.getId(), self.getBirthday(), "  ", self.getDepartment())


```

```python
from algorithmsinpython.chapter2.Person import  Person
from algorithmsinpython.chapter2.Student import Student

if __name__ == "__main__":
    student = Student("王", "男", (2018, 11, 1), "1607094155", "146")
    student.show()
    Person.show()
    print(Person.person_number)
    super(Student,student).showtest()
    
```

```text
super有两种用法：
①：在子类中直接用super().m()   #会向上搜索m函数，进行调用（super中没有参数）
②：在任意地方使用：super(C,obj).m() #会从C类开始向上搜索（包括C），搜索到了m函数后就使用obj作为self调用
```

* #### 重构系统运算符

  ```python
  class Person:
      @staticmethod
      def show():
          print("show")

      person_number = 0

      def __init__(self, strname, strsex, tuplebirthday, strid):
          self._name = strname
          self._sex = strsex
          self._birthday = tuplebirthday
          self._id = strid
          Person.show()
          Person.person_number += 1

      def getName(self):
          return self._name

      def getSex(self):
          return self._sex

      def getBirthday(self):
          return self._birthday

      def getId(self):
          return self._id

      def showtest(self):
          print("im super")
  ```


  	"""
  	重构等号
  	"""
      def __eq__(self, other):
          return self.getId() == other.getId()


      """
      print(person == person2)
      """
  ```

  ```text
  python为所有的运算符规定了特殊方法名，所有特殊的名字都以两个下划线开始，再以两个下划线结束
  ```

* #### python异常

```python
#可以自己抛出
if dum == 0:
    raise ZeroDivisionError
    
#也可以try except
try:
    print("hello")
except TypeError:
    print("error")
```

## chapter 3

### list

* 用法和数组类似

  注意：sort函数只有当内部元素类型相同时才能使用

### 链表

```python
def testList():
    """
    注意这里，list必须声明成global，或者拿到方法外部进行声明，
    否则会出现“UnboundLocalError： local variable 'xxx' referenced before assignment”
    :return:
    """
    global list
    list = list()
    num = 1
    list.append("111")
    list.append(2)
    list.append((2018, 11, 1))
    print(list)
    print(['111', 2, (2018, 11, 1)])
    print(list[2:])
    print(num)


class LNode:
    """
    结点，elem用来保存当前的值，next用来存放下一个LNode
    这里要理解：python中变量名相当于就是指针！方法名也是指针，如果想对同一个函数中的某个变量进行不同的操作可以
    将函数名作为参数进行传递
    """

    def __init__(self, elem, next_=None):
        self._elem = elem
        self._next = next_

    def getElem(self):
        return self._elem

    def getNext(self):
        return self._next


class MyList:
    def __init__(self, elem):
        self._head = LNode(elem)

    def isEmpty(self):
        if self._head is None:
            return True
        else:
            return False

    def append(self, elem):
        temp = self._head
        while temp.getNext() is not None:
            temp = temp.getNext()
        temp._next = LNode(elem)
        print(self._head._next.getNext())  # 输出为<__main__.LNode object at 0x0000014F6F72A278>

    def appendOnLocation(self, elem, location):
        temp = self._head
        count = 1
        while count != location - 1:
            temp = temp.getNext()
            count += 1
        newItem = LNode(elem)
        newItem._next = temp.getNext()
        temp._next = newItem

    def delete(self, index):
        p = self._head
        isHead = True
        prev = self._head
        next = self._head.getNext()
        while p is not None:
            if (p.getElem() == index):

                if isHead:
                    self._head = next
                else:
                    isHead = False
                    prev._next = next
                break
            else:
                isHead = False
                prev = p
                p = next
                next = p.getNext()

    def showList(self, proc):
        p = self._head
        while p is not None:
            proc(p.getElem(), "\t")
            p = p.getNext()

    def elements(self):
        p = self._head
        while p is not None:
            yield p.getElem()
            p=p.getNext()


if __name__ == "__main__":
    # testList()
    list = MyList(-1)
    for i in range(10):
        list.append(i)
    list.showList(print)
    list.delete(2)
    print("---------------------------------")
    list.showList(print)
    list.appendOnLocation(55, 4)
    print("---------------------------------")
    list.showList(print)
    print("---------------------------------")
    for x in list.elements():
        print(x)
```

* #### python中变量名的性质

```text
python中变量名相当于就是指针！方法名也是指针，如果想对同一个函数中的某个变量进行不同的操作可以
将函数名作为参数进行传递，如showList()方法，传递函数的使用可以结合遍历构成筛选器（filter）,将筛选条件“打包”传入
```

* ####  yield关键字 

```text
如上代码中elements(self)函数所示，使用yield关键字可以将当前的变量存到一个临时的数组里供for循环使用
```

## chapter4

* #### 正则表达式

```text
re包:
r1 = re.compile("a*b*c")  #正则表达式对象
然后可以通过调用re包内的一些方法进行使用（正则表达式对象参数名为pattern）

```

##### 一些常见的特殊符号：.   ^   $   *   +	?	\	|	{	}	[	]	(	)

> ## .
>
> ```txt
> 通配符
> ```
>
> ```python
> import re
> r1 = re.compile("a.bc")
> testText = "hello my name is akbc and my sister's name is aabbc aaakbbbc"
> match_r1 = re.findall(pattern=r1, string=testText)
> print(match_r1)
>
> """
> ['akbc', 'abbc']
> """
> ```
>
> ## ^    $
>
> ```text
> 首尾描述符，^为行首描述符，$为行尾描述符 ， ^与[]结合时是取反
> ```
>
> ```python
> import re
>
> # 裘宗燕的《数据结构与算法Python语言描述》源码128页
> print(re.search('^for', 'books\nfor children'))
> print(re.search("like$", "cats like\nto eat fishes"))
> print(re.findall("^for", 'for books\nfor children'))
> print(re.findall(".*like$", "cats like\nto eat fishes aslike"))
> print(re.findall(".*like$", "cats like\nto eat fishes aslike",re.M))
> #re.M 设置为多行读取
> """
> None
> None
> ['for']
> ['to eat fishes aslike']
> ['cats like', 'to eat fishes aslike']
> """
> ```
>
> ## *    +
>
> ```text
> 重复符 , *为任意重复数量（0也行），+为大于等于1的次数
> ```
>
> ```python
> re1 = re.compile("a*bc")
> print(re.findall(re1, "abc and bc and aaabc and aaaaabcc"))
> """
> ['abc', 'bc', 'aaabc', 'aaaaabc']
> """
> re1 = re.compile("a+bc")
> print(re.findall(re1, "abc and bc and aaabc and aaaaabcc"))
> """
> ['abc', 'aaabc', 'aaaaabc']
> """
> ```
>
> ## ?
>
> ```text
> 可选描述符
> ```
>
> ```python
> re1 = re.compile("a?bc")
> print(re.findall(re1, "abc and bc and aaabc and aaaaabcc"))
> """
> ['abc', 'bc', 'abc', 'abc']
> """
> ```
>
> ## \
>
> ```text
> 转义字符
> 常用的：
> 取特殊字符的原本意义：\^   \*   \\ 等
> \d:十进制数字，介于[0-9]
> \D:非十进制数字，介于[^0-9]
> \s:与所有空白字符匹配，等价于[\t\v\b\f\r]
> \S:与所有非空白字符匹配,等价于[^\t\v\b\f\r]
> \w:与所有字母、数字字符匹配，等价于[0-9a-zA-Z]
> \W:与所有非字母数字字符匹配，等价于[^0-9a-zA-Z]
> \t:tab
> \n：换行
> \b:单词边界 （但是python字符串中\b是退格符，所以要使用边界描述就必须使用原始字符--在string前加r）
>
> ```
>
> ```python
> re1 = re.compile(r"\babc\b")
> print(re.findall(re1, "abc and bc and aaabc and aaaaabcc"))
> """
> ['abc']
> """
> ```
>
> ## |
>
> ```text
> 选择描述符（或）
> ```
>
> ```python
> re1 = re.compile(r"abc|and")
> print(re.findall(re1, "abc and bc and aaabc and aaaaabcc"))
> """
> ['abc', 'and', 'and', 'abc', 'and', 'abc']
> """
> ```
>
> ## { }
>
> ```python
> 重数次数的范围描述符
> ```
>
> ```python
> re1 = re.compile("a{2,3}bc")
> print(re.findall(re1, "abc and bc and aaabc and aaaaabcc"))
> """
> ['aaabc', 'aaabc']
> """
> ```
>
> ## [ ]
>
> ```text
> 字符组（用来描述一个字符）
> ```
>
> ```python
> re1 = re.compile("a[ksd]bc")   #[]处的字符可以为k、s或d
> print(re.findall(re1, "akbc and asbc and aaabc and aaaaabcc"))
> """
> ['akbc', 'asbc']
> """
> re1 = re.compile("a[0-9A-Z]bc")#0-9或A-Z
> print(re.findall(re1, "a1bc and asbc and aaaRbc and aaaaabcc"))
> """
> ['a1bc', 'aRbc']
> """
>
> re1 = re.compile("a[^0-9A-Z]bc")#非0-9或A-Z
> print(re.findall(re1, "a1bc and asbc and aaaRbc and aaaaabcc"))
> """
> ['asbc', 'aabc']
> """
> ```
>
> ## ( )
>
> ```text
> 组
> ```
>
> ```python
> re1 = re.compile(".((.)e)f")  #模式中(pattern)各对括号按开括号的顺序编号
> print(re.findall(re1, "abcdef"))
> """
> [('de', 'd')]   #之所以得到的是元组，是因为上面开了两次括号
> """
> re1 = re.compile(r"(.{2}) \1")  #\n应用前面匹配成功的值（相当于复制前面的值），n是元组中的第几个，注意这里是从1开始，而不是0
> print(re.findall(re1, "oh oh no oh"))
> """
> ['oh']
> """
> ```
>
> 