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
from termcolor import colored as cl # text customization

# Exploring & analyzing the dataset - Изследване и анализиране на данните
input = pd.read_csv("./dataset/creditcard.csv")
input.dropna(thresh=284315)
data = input
print(data.sample(frac=0.1).head(n=5))
# From the above it could be infered that the dataset has 
# 28 anonymized features and 2 non-anonymized features:
# 1. Amount and 2. Class (whether the transcation was a fraud or not)

# От горното можем да заключим, че данните представляват
# 28 анонимни и 2 именувани полета:
# 1. Стойност и 2. Клас (описващ дали транзакцията е била измама или не)

# Dataset distribution
# Разпределение на данните
print(data.describe())
negatives = data[data['Class']==0]
positives = data[data['Class']==1]
fraud_percentage = (len(positives)/len(input))*100
# It can infered that the datset is skewed with just 0.17274% 
# fradulent examples. One could simply get a overall accuracy 
# of 99.82726% by predicting every example isn't a fradulent example. 
# But, the approach does not solve the problem. So recall rate 
# (True positives / (True positives + False negatives)) would be the metric to optimize.

# Данните са "изкривени" със само 0.17274% примери на измама.
# С такъв тип данни лесно можем да получим 99.82726% точност,
# ако предположим, че всеки тестов пример е валиден (не е измама).
# Но това не е решение на задачата. Ще опитаме да оптимизираме метриката
# True positives / (True positives + False negatives).

print(cl('CASE COUNT', attrs = ['bold']))
print(cl('--------------------------------------------', attrs = ['bold']))
print(cl('Total number of cases are {}'.format(len(data)), attrs = ['bold']))
print(cl('Number of Non-fraud cases are {}'.format(len(negatives)), attrs = ['bold']))
print(cl('Number of Non-fraud cases are {}'.format(len(positives)), attrs = ['bold']))
print(cl('Percentage of fraud cases is {}'.format(fraud_percentage), attrs = ['bold']))
print(cl('--------------------------------------------', attrs = ['bold']))

print(cl('CASE AMOUNT STATISTICS', attrs = ['bold']))
print(cl('--------------------------------------------', attrs = ['bold']))
print(cl('NON-FRAUD CASE AMOUNT STATS', attrs = ['bold']))
print(negatives.Amount.describe())
print(cl('--------------------------------------------', attrs = ['bold']))
print(cl('FRAUD CASE AMOUNT STATS', attrs = ['bold']))
print(positives.Amount.describe())
print(cl('--------------------------------------------', attrs = ['bold']))

# To further observe if there is any sort of correlation between the various 
# parameters of the dataset, a correlation matrix could be built. This matrix 
# gives a great idea of whether there is any strong correlation or no correlation 
# (i.e. could be removed) or if there is any semblance of linear correlation.

# Correlation matrix
correlation_matrix = data.corr()
fig = plt.figure(figsize = (12,9))

sns.heatmap(correlation_matrix,vmax = 0.4, square = True)
plt.savefig("./plot/correlation.png")

# Thus, from the correlation matrix heat map above, most of the V parameters do not 
# have any correlation with each other. However, parameters V1 to V18 show pretty 
# strong correlation with the class parameter and all remaining parameters show nearly
# no correlation with the class column. Also, some V parameters show strong positive 
# correlation and some others show strong negative correlation.

# So, the question now arises whether the V parameters not having strong correlation 
# with the Class parameter influences the machine learning model or not. So in this 
# project, both possibilities of training the model with all the V parameters and also 
# with only V parameters from V1 to V18 were entured.