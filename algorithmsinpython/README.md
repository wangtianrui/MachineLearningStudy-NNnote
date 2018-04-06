Algorithms In Python

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

