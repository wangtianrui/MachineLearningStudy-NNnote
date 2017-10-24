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