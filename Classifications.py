from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import random

random.seed(54)
def RandomForest(x_train,x_valid,y_train):
    randomforest = RandomForestClassifier(random_state=1)
    randomforest.fit(x_train, y_train)
    y_pred = randomforest.predict(x_valid)

    return y_pred,randomforest

def DecisionTree(x_train,x_valid,y_train):
    decisiontree = DecisionTreeClassifier(random_state=1)
    decisiontree.fit(x_train, y_train)
    y_pred = decisiontree.predict(x_valid)

    return y_pred,decisiontree

    
def GradientBoosting(x_train,x_valid,y_train):
    gbk = GradientBoostingClassifier(random_state=1)
    gbk.fit(x_train, y_train)
    y_pred = gbk.predict(x_valid)

    return y_pred,gbk


def NaiveBayes(x_train,x_valid,y_train):
    gaussian = GaussianNB()
    gaussian.fit(x_train, y_train)
    y_pred = gaussian.predict(x_valid)

    return y_pred,gaussian
    
