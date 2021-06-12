from PIL import Image
import re

import sys
assert sys.version_info >= (3, 5)

import sklearn
assert sklearn.__version__ >= "0.20"


# Common imports
import numpy as np
import os



lable_pt1 = "_[0-9]"                #Nomal's labe = 0     Pneumonia's lable = 1 
lable_pt2 = re.compile(lable_pt1)

path_dir = "/home/j/MLCourse/final/data_npy/"
file_list = os.listdir(path_dir)

y_train = np.array([])
X_train = np.empty((0,6144))   #6144 = Width * Height * Channel

for jpeg in file_list:
    image = np.load(path_dir + jpeg)
    X_train = np.append(X_train,image.reshape(-1,64*32*3),axis = 0)
    y_train = np.append(y_train,int(lable_pt2.search(jpeg).group()[1:]))



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,test_size=0.2,stratify = y_train, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,stratify = y_train,random_state=42)

from sklearn.svm import LinearSVC

svm_clf = LinearSVC(max_iter=1000, tol = 20 ,C = 0.000001,random_state=42)
svm_clf.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
print("cross_val score : " ,cross_val_score(svm_clf, X_train, y_train, cv=5, scoring="accuracy").mean())
print("test score : ",svm_clf.score(X_test, y_test))

