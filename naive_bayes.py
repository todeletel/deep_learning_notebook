import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

X = np.array([[1, 'S'],
              [1, 'M'],
              [1, 'M'],
              [1, 'S'],
              [1, 'S'],
              [2, 'S'],
              [2, 'M'],
              [2, 'M'],
              [2, 'L'],
              [2, 'L'],
              [3, 'L'],
              [3, 'M'],
              [3, 'M'],
              [3, 'L'],
              [3, 'L'],
              ])


y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
data = pd.DataFrame(X)


lb_model = LabelEncoder()
data[1] = lb_model.fit_transform(data[1])

data[2] = y

X = data.iloc[:, 0:2]
y = data.iloc[:, -1]
""" 
# CountVectorizer().fit_transform(data)
naive_bayes = GaussianNB()
naive_bayes.fit(data, y.ravel())
t = lb_model.transform(['S'])
rs = naive_bayes.predict(np.array([[2, t[0]]]))
print("GaussianNB predict:", rs)

naive_bayes = MultinomialNB()
naive_bayes.fit(data, y.ravel())
t = lb_model.transform(['S'])
rs = naive_bayes.predict(np.array([[2, t[0]]]))
print("MultinomialNB predict:", rs)

naive_bayes = BernoulliNB()
naive_bayes.fit(data, y.ravel())
t = lb_model.transform(['S'])
rs = naive_bayes.predict(np.array([[2, t[0]]]))
print("BernoulliNB predict:", rs)
"""

print(data)

def customBayes(X, y):
    y_dict = dict()
    for y_one in y:
        if y_one in y_dict:
            y_dict[y_one] = y_dict[y_one] + 1.0
        else:
            y_dict[y_one] = 1.0
    for key, value in y_dict.items():
        y_dict[key] = value / len(y)
    print(y_dict)


customBayes(X, y)
#print(list(y))