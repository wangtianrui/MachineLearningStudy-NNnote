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

#### 2-Iris-KNeighbors.py

>利用预设值K，对样本进行分类，该模型没有训练过程

```txt
(150, 4)
Iris Plants Database
====================

Notes
-----
Data Set Characteristics:
    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica
    :Summary Statistics:

    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20  0.76     0.9565  (high!)
    ============== ==== ==== ======= ===== ====================

    :Missing Attribute Values: None
    :Class Distribution: 33.3% for each of 3 classes.
    :Creator: R.A. Fisher
    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
    :Date: July, 1988

This is a copy of UCI ML iris datasets.
http://archive.ics.uci.edu/ml/datasets/Iris

The famous Iris database, first used by Sir R.A Fisher

This is perhaps the best known database to be found in the
pattern recognition literature.  Fisher's paper is a classic in the field and
is referenced frequently to this day.  (See Duda & Hart, for example.)  The
data set contains 3 classes of 50 instances each, where each class refers to a
type of iris plant.  One class is linearly separable from the other 2; the
latter are NOT linearly separable from each other.

References
----------
   - Fisher,R.A. "The use of multiple measurements in taxonomic problems"
     Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
     Mathematical Statistics" (John Wiley, NY, 1950).
   - Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis.
     (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
   - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
     Structure and Classification Rule for Recognition in Partially Exposed
     Environments".  IEEE Transactions on Pattern Analysis and Machine
     Intelligence, Vol. PAMI-2, No. 1, 67-71.
   - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
     on Information Theory, May 1972, 431-433.
   - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
     conceptual clustering system finds 3 classes in the data.
   - Many, many more ...

Accuracy: 0.894736842105
             precision    recall  f1-score   support

     setosa       1.00      1.00      1.00         8
 versicolor       0.73      1.00      0.85        11
  virginica       1.00      0.79      0.88        19

avg / total       0.92      0.89      0.90        38

```
#### 2-Titanic-Tree.py

```txt
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1313 entries, 0 to 1312
Data columns (total 3 columns):
pclass    1313 non-null object
age       1313 non-null float64
sex       1313 non-null object
dtypes: float64(1), object(2)
memory usage: 30.9+ KB
None
['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male']
Accuracy: 0.781155015198
             precision    recall  f1-score   support

       died       0.78      0.91      0.84       202
   survived       0.80      0.58      0.67       127

avg / total       0.78      0.78      0.77       329

```

#### 2-Titanic-Forest.py

```txt
Accuracy: 0.781155015198
             precision    recall  f1-score   support

          0       0.78      0.91      0.84       202
          1       0.80      0.58      0.67       127

avg / total       0.78      0.78      0.77       329

Accuracy: 0.775075987842
             precision    recall  f1-score   support

          0       0.77      0.91      0.83       202
          1       0.79      0.57      0.66       127

avg / total       0.78      0.78      0.77       329

Accuracy: 0.790273556231
             precision    recall  f1-score   support

          0       0.78      0.92      0.84       202
          1       0.82      0.58      0.68       127

avg / total       0.80      0.79      0.78       329

```
>集成分类器，各个分类器之间相互作用