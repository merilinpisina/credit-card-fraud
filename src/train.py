import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
import test as testing

input = pd.read_csv("./dataset/creditcard.csv")
input.dropna(thresh=284315)
data = input[:50000]

negatives = data[data['Class']==0]
positives = data[data['Class']==1]

# One Class SVM (Support Vector Mashine)
# Метод на опорните вектори - One Class SVM (Support Vector Mashine).
# Този алгоритъм представя обучаващите примери като точки в n-мерно пространство.
# Примерите са проектират в пространството по такъв начин, че да бъдат линейно разделими.

# RBF Kernel (non-linear)
svmRBF = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
svmRBF.fit(negatives)

# Linear Kernel
svmLinear = svm.OneClassSVM(nu=0.1, kernel="linear", gamma=0.1)
svmLinear.fit(negatives)

# Isolation Forest
isolationForest = IsolationForest()
isolationForest.fit(negatives)

train_linear = svmLinear.predict(negatives)
test_linear = svmLinear.predict(positives)

train_isolation_forest = isolationForest.predict(negatives)
test_isolation_forest = isolationForest.predict(positives)

train_RBF = svmRBF.predict(negatives)
test_RBF = svmRBF.predict(positives)

print("Training: One Class SVM (RBF) : ",(testing.train_accuracy(train_RBF)),"%")
print("Test: One Class SVM (RBF) : ",(testing.test_accuracy(test_RBF)),"%")

print("Training: Isolation Forest: ",(testing.train_accuracy(train_isolation_forest)),"%")
print("Test: Isolation Forest: ",(testing.test_accuracy(test_isolation_forest)),"%")

print("Training: One Class SVM (Linear) : ",(testing.train_accuracy(train_linear)),"%")
print("Test: One Class SVM (Linear) : ",(testing.test_accuracy(test_linear)),"%")

# From the above analysis it could be noted that Isolation
# Forest does the best among Anomaly detection algorithms.