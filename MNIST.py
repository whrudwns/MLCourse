import time

start= time.time()

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

import numpy as np


# Common imports
import numpy as np
import os

####fetch MNIST###
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)


for rand in [10,100,1000,10000]:
    print("\n###### rand_seed and random_state = ",rand," ######\n")
    np.random.seed(rand)
    
    
    ###divide data to train and test###
    
    from sklearn.model_selection import train_test_split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        mnist.data, mnist.target, test_size=10000, random_state=rand)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=10000, random_state=rand)
    
    
    ###call models###
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    
    random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=rand)
    extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=rand)
    svm_clf = LinearSVC(max_iter=100, tol=20, random_state=rand)
    mlp_clf = MLPClassifier(random_state=rand)
    
    estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
    for estimator in estimators:
        print("Training the", estimator)
        s_time = time.time()
        estimator.fit(X_train, y_train)
        e_time = time.time()
        print("elapsed time : ", e_time - s_time,"\n")
    
    
    for estimator in estimators:
        print("\n{} validation score : ".format(estimator), estimator.score(X_val, y_val))
    
    ###ensemble###
    from sklearn.ensemble import VotingClassifier
    named_estimators = [
        ("random_forest_clf", random_forest_clf),
        ("extra_trees_clf", extra_trees_clf),
        ("svm_clf", svm_clf),
        ("mlp_clf", mlp_clf),
    ]
    voting_clf = VotingClassifier(named_estimators)
    s_time = time.time()
    voting_clf.fit(X_train, y_train)
    e_time = time.time()
    print("\n voting_clf training time : ",e_time - s_time)
    print("\n[validation]  hard voting_clf score : ",voting_clf.score(X_val, y_val))

    print("\n[test]  hard voting_clf score : ",voting_clf.score(X_test, y_test))
    
    ###remove SVM###
    voting_clf.set_params(svm_clf=None)
    voting_clf.estimators
    
    del voting_clf.estimators_[2]   
    
    print("\n[validation]  hard voting_clif without SVM : ",voting_clf.score(X_val, y_val))
    
    voting_clf.voting = "soft"
    print("\n[validation]  soft voting_clif without SVM : ",voting_clf.score(X_val, y_val))
    
    
    voting_clf.voting = "hard"
    print("\n[test]  hard voting_clif without SVM : ",voting_clf.score(X_test, y_test))
    
    end = time.time()
    
    print("\n total time elapsed(s) : ",(end-start))
