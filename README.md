# MachineLearningStudy

#### 2-BreastCancer-logistic&SGD.py

>线性分类器（考虑所有样本）

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

>线性分类器（考虑代表性样本）

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


#### 2-News-Bayes.py

>朴素贝叶斯分类器，单独考虑每一个维度的特征（大多用于文本分类）

```txt
18846
From: Mamatha Devineni Ratnam <mr47+@andrew.cmu.edu>
Subject: Pens fans reactions
Organization: Post Office, Carnegie Mellon, Pittsburgh, PA
Lines: 12
NNTP-Posting-Host: po4.andrew.cmu.edu



I am sure some bashers of Pens fans are pretty confused about the lack
of any kind of posts about the recent Pens massacre of the Devils. Actually,
I am  bit puzzled too and a bit relieved. However, I am going to put an end
to non-PIttsburghers' relief with a bit of praise for the Pens. Man, they
are killing those Devils worse than I thought. Jagr just showed you why
he is much better than his regular season stats. He is also a lot
fo fun to watch in the playoffs. Bowman should let JAgr have a lot of
fun in the next couple of games since the Pens are going to beat the pulp out of Jersey anyway. I was very disappointed not to see the Islanders lose the final
regular season game.          PENS RULE!!!


Accuracy： 0.839770797963
                          precision    recall  f1-score   support

             alt.atheism       0.86      0.86      0.86       201
           comp.graphics       0.59      0.86      0.70       250
 comp.os.ms-windows.misc       0.89      0.10      0.17       248
comp.sys.ibm.pc.hardware       0.60      0.88      0.72       240
   comp.sys.mac.hardware       0.93      0.78      0.85       242
          comp.windows.x       0.82      0.84      0.83       263
            misc.forsale       0.91      0.70      0.79       257
               rec.autos       0.89      0.89      0.89       238
         rec.motorcycles       0.98      0.92      0.95       276
      rec.sport.baseball       0.98      0.91      0.95       251
        rec.sport.hockey       0.93      0.99      0.96       233
               sci.crypt       0.86      0.98      0.91       238
         sci.electronics       0.85      0.88      0.86       249
                 sci.med       0.92      0.94      0.93       245
               sci.space       0.89      0.96      0.92       221
  soc.religion.christian       0.78      0.96      0.86       232
      talk.politics.guns       0.88      0.96      0.92       251
   talk.politics.mideast       0.90      0.98      0.94       231
      talk.politics.misc       0.79      0.89      0.84       188
      talk.religion.misc       0.93      0.44      0.60       158

             avg / total       0.86      0.84      0.82      4712
```