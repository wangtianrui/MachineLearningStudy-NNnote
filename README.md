# MachineLearningStudy

#### 2-BreastCancer-logistic&SGD.py
```txt
(683, 11)
2    344
4    168
Name: Class, dtype: int64
2    100
4     71
Name: Class, dtype: int64


Accuracy : 0.988304093567
             precision    recall  f1-score   support

     Benign       0.99      0.99      0.99       100
  Malignant       0.99      0.99      0.99        71

avg / total       0.99      0.99      0.99       171


Accuracy : 0.964912280702
             precision    recall  f1-score   support

     Benign       0.98      0.96      0.97       100
  Malignant       0.95      0.97      0.96        71

avg / total       0.97      0.96      0.96       171
```
>logistic是以极大似然来进行参数的优化

>SGD则是由求导进行梯度训练


#### 2-Number-SVM.py

```txt
(1797, 64)
(1347, 64) (450, 64) (1347,) (450,)
Accuracy : 0.953333333333
             precision    recall  f1-score   support

          0       0.92      1.00      0.96        35
          1       0.96      0.98      0.97        54
          2       0.98      1.00      0.99        44
          3       0.93      0.93      0.93        46
          4       0.97      1.00      0.99        35
          5       0.94      0.94      0.94        48
          6       0.96      0.98      0.97        51
          7       0.92      1.00      0.96        35
          8       0.98      0.84      0.91        58
          9       0.95      0.91      0.93        44

avg / total       0.95      0.95      0.95       450
```

>相对于logistic与SDC，SVM是通过两个样本的位子来确定分类器，向量的位置不是受所有样本影响，而是两个空间间隔最小的样本