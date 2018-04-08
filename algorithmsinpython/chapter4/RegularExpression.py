import re

# 裘宗燕的《数据结构与算法Python语言描述》源码128页
"""
print(re.search('^for', 'books\nfor children'))
print(re.search("like$", "cats like\nto eat fishes"))
print(re.findall("^for", 'for books\nfor children'))

re1 = re.compile("a*bc")
print(re.findall(re1, "abc and bc and aaabc and aaaaabcc"))
re1 = re.compile("a+bc")
print(re.findall(re1, "abc and bc and aaabc and aaaaabcc"))
re1 = re.compile("a?bc")
print(re.findall(re1, "abc and bc and aaabc and aaaaabcc"))
print(re.search('^for', 'books\nfor children',re.M))
print(re.findall(".*like$", "cats like\nto eat fishes aslike",re.M))
"""
re1 = re.compile(".((.)e)f")
print(re.findall(re1, "abcdef"))

re1 = re.compile(r"(.{2}) \1")
print(re.findall(re1, "oh oh no oh"))