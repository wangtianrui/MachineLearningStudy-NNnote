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
            p = p.getNext()

    def filter(self, pred):
        p = self._head
        while p is not None:
            if pred(p.getElem()):
                yield p.getElem()
                p = p.getNext()

    def rev(self):
        prev = None
        while self._head is not None:
            temp = self._head
            self._head = temp.getNext()
            temp._next = prev
            prev = temp
        self._head = prev


"""
 def sort1(self):
     if self._head is None:
         return
     now = self._head.getNext()
     while now is not None:
         x = now.getElem()
         prev = self._head
         while prev is not now and prev.getElem() <= x:
             prev = prev.getNext()
         while prev is not now:
             temp = prev.getElem()
             x = temp
             prev = prev.getNext()
         now._elem = x
         now = now.getNext()
"""


def sort(self):
    prev = self._head
    if prev is None or prev.getNext() is None:
        return
    rem = prev.getNext()
    prev._next = None
    while rem is not None:
        prev = self._head
        q = None
        while prev is not None and prev.getElem() <= rem.getElem():
            q = prev
            prev = prev.getNext()
        if q is None:
            self._head = rem
        else:
            q._next = rem
        q = rem
        rem = rem.getNext()
        q._next = prev


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
    list.rev()
    print("---------------------------------")
    list.showList(print)
    list.sort()
    print("---------------------------------")
    list.showList(print)
