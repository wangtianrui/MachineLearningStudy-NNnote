from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

news = fetch_20newsgroups(subset='all')