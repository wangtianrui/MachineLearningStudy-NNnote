import pandas as pd

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
print(titanic.head())

X = titanic[['pclass', 'age', 'sex']]
Y = titanic['survived']

print(X.info())

X['age'].fillna(X['age'].mean(), inplace=True)
print(X.info())
