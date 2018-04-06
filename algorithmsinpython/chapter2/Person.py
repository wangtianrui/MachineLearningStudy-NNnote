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


    def __eq__(self, other):
        return self.getId() == other.getId()