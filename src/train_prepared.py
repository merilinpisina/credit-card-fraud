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
from sklearn.metrics import classification_report, accuracy_score
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import test as testing

# Data Preparation and ML Algorithm training

# In order to prepare the data for the machine learning models, the data is filtered 
# and split into following subsets: A subset containing all the V parameters except 
# for Class, a subset containing only those V parameters which have a strong 
# correlation with the Class label and the subset coontaining the Claass column.

# Then the isolation forest model and the local outlier factor algorithm were 
# chosen and fit to both the input parameters subsets in sequence.

# Splitting the data set into training set ==> all the parameters (or only correlating parameters) 
# i.e. the training and testing set, and  the evaluating column Class.

# To extract columns from  the datframe.
input = pd.read_csv("./dataset/creditcard.csv")
input.dropna(thresh=284315)
data = input[:50000]
count_fraud_trans = data['Class'][data['Class'] == 1].count()
count_valid_trans = data['Class'][data['Class'] == 0].count()
percent_outlier = count_fraud_trans/(count_valid_trans)
negatives = data[data['Class']==0]
positives = data[data['Class']==1]

columns = data.columns.tolist()

# Filtering the columns as required
# 1. All V parameters and excluding class
columns_V_all = [c for c in columns if c not in ["Class"]]

# 2. Some V parameters which are correlating with Class and excluding class and  amount columns
columns_V_part = [c for c in columns_V_all if c not in ["Class", "Amount", "V22","V23", "V24", "V25", "V26", "V27", "V28"]]


#3 Evaluating colunmn Class
col_eval = data["Class"]

X_types = [columns_V_all, columns_V_part]

for x in  X_types:
    
    models = {"LOF": LocalOutlierFactor(n_neighbors= 20,  contamination = percent_outlier),
          "IsF": IsolationForest(max_samples = len(data[x]),  contamination = percent_outlier, random_state = 1)}
    print(len(x))
    keys = list(models.keys())
    
    if  keys[0]=="LOF":
        mod_name = "LOF"
        model = models.get("LOF")
        Y_pred = model.fit_predict(data[x])
        scores_pred = model.negative_outlier_factor_
         ## The  prediction value for these models by default give -1 and +1 which needs to be changed to 0 and  1
        Y_pred[Y_pred == 1] =0
        Y_pred[Y_pred == -1] =1    
        error_count = (Y_pred != col_eval).sum()
        # Printing the metrics for the classification algorithms
        print('{}: Number  of errors {}'.format(mod_name, error_count))
        print("accuracy score: ", accuracy_score(col_eval,Y_pred))
        print(classification_report(col_eval,Y_pred))
        
        
    if keys[1] =="IsF":
        mod_name = "IsF"
        model = models.get("IsF")
        model.fit(data[x])
        scores_pred = model.decision_function(data[x])
        Y_pred = model.predict(data[x])
        ## The  prediction value for these models by default give -1 and +1 which needs to be changed to 0 and  1
        Y_pred[Y_pred == 1] =0
        Y_pred[Y_pred == -1] =1
        error_count = (Y_pred != col_eval).sum()        
        # Printing the metrics for the classification algorithms
        print('{}: Number  of errors {}'.format(mod_name, error_count))
        print("accuracy score: ", accuracy_score(col_eval,Y_pred))
        print(classification_report(col_eval,Y_pred))

# A The accuracy scores for both the models are seen to be about 99.7% which is 
# extremely high which implies that the model is performing with high accuracy. 
# However, from the number of errors in the predictions sfor both the models seem 
# high too implying the contrary. This is because majority population of the 
# transactions are valid thereby making the accuracy which is the sum of both true 
# positives and true negatives over the total data points, a biased metric. In such 
# cases, precision and the F1 score give a better measure of the performance of a 
# model. Thus, in case of the model using all the V parameters the precision of 
# Local Outlier Factors algorithm is obly about 5 % precise where as the Isolation 
# Forest algorithm showed about 34%. While 30% precise model is not a great model 
# but of the two models isolation forest model performs the best.

# Also, by reducing the dimensionality of the V parameters a marked increase in 
# precision was observed at 14 % for the Local Outlier Factor algorithm and about 
# 39% precise. Thus, a further increase in the precision of the models could be 
# obtained by doing some feature engineering.

# Some other ML models could be investigated.