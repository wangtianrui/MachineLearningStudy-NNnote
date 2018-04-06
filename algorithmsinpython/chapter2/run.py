from algorithmsinpython.chapter2.Person import Person
from algorithmsinpython.chapter2.Student import Student

if __name__ == "__main__":
    student = Student("王", "男", (2018, 11, 1), "1607094155", "146")
    student.show()
    Person.show()
    print(Person.person_number)
    person = Person("王", "男", (2018, 11, 1), "1607094155")
    super(Student, student).showtest()
    person2 = Person("王", "男", (2018, 11, 1), "1607094151")
    print(person == person2)
