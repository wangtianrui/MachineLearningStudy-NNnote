from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

news = fetch_20newsgroups(subset='all')
x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

"""
#使用CountVectorizer
count_vec = CountVectorizer()
x_train = count_vec.fit_transform(x_train)
x_test = count_vec.transform(x_test)
print(x_train)

"""
tfidf_vec = TfidfVectorizer()
x_train = tfidf_vec.fit_transform(x_train)
x_test = tfidf_vec.transform(x_test)
print(x_train)
