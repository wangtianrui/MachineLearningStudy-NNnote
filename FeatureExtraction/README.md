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

## 文本特征提取

* #### CountVectorizer & TfidfVectorizer

```txt
CountVectorizer用于在高维词汇向量空间上画出每个样本的频率（不同单词出现的次数）


TfidfVectorizer在CountVectotizer的基础上加入一个衡量值，衡量值=出现次数的倒数，可以将停用词（在不同文章中出现次数都比较多）的影响大大降低
```