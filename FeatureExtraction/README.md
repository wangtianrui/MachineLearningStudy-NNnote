## DictVectorizer

>可将字典转换成矩阵

```python
measurements = [{'city': 'Dubai', 'temperature': 33.},
                {'city': 'London', 'temperature': 12.},
                {'city': 'San Fransisco', 'temperature': 18.}]
vec = DictVectorizer()
array = vec.fit_transform(measurements).toarray()
```

```txt
[[  1.   0.   0.  33.]
 [  0.   1.   0.  12.]
 [  0.   0.   1.  18.]]
['city=Dubai', 'city=London', 'city=San Fransisco', 'temperature']
  (0, 0)	1.0
  (0, 3)	33.0
  (1, 1)	1.0
  (1, 3)	12.0
  (2, 2)	1.0
  (2, 3)	18.0
```

