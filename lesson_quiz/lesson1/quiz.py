import pandas
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, fbeta_score, roc_auc_score, r2_score, mean_squared_error

df = pandas.read_csv('2_class_data.csv')
X = np.array(df[['x1', 'x2']])
y = np.array(df['y'])


classifier_factory = {
    "LogisticRegression": LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier(),
    "SVC": SVC()
}

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

for key, classifier in classifier_factory.items():
    classifier.fit(train_X, train_y)
    predict_y = classifier.predict(train_X)
    beta = 0.01
    print("\n----------------", key, "----------------")
    # print("f1_score:", f1_score(y_true=train_y, y_pred=predict_y))
    # print("f_beta_score:", fbeta_score(y_true=train_y, y_pred=predict_y, beta=beta))
    # print("precision:", precision_score(y_true=train_y, y_pred=predict_y))
    # print("recall_score:", recall_score(y_true=train_y, y_pred=predict_y))
    test_predict_y = classifier.predict(test_X)
    print("test_f1_score:", f1_score(y_true=test_y, y_pred=test_predict_y))
    print("test_f_beta_score:", fbeta_score(y_true=test_y, y_pred=test_predict_y, beta=beta))
    print("test_precision:", precision_score(y_true=test_y, y_pred=test_predict_y))
    print("test_recall_score:", recall_score(y_true=test_y, y_pred=test_predict_y))

    # print("roc_auc_score:", roc_auc_score(y_true=train_y,y_pred=predict_y)


