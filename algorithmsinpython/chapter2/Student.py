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

