import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('https://raw.githubusercontent.com/omairaasim/machine_learning/master/project_11_k_nearest_neighbor/iphone_purchase_records.csv')
print(df)
print(df.Gender.value_counts())

print(df.loc[df['Purchase Iphone']==1,"Gender"].value_counts())

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Label Encoding
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
X.Gender = enc.fit_transform(X.Gender)


# Spliting the data into sets
skf = StratifiedKFold(n_splits=5)
for train_index,test_index in skf.split(X,y):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Feature Scaling
scale= StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)

# Model selection
log = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=5)

# Training the model
log.fit(X_train, y_train)
knn.fit(X_train, y_train)


# Test the model
y_log_pred = log.predict(X_test)
y_knn_pred = knn.predict(X_test)
newdf = pd.DataFrame({"Actual":y_test, "Predicted":y_knn_pred})
print(newdf.head())

print(confusion_matrix(y_test, y_knn_pred))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_log_pred))



#Hyper Parameter Tuning
lis = [i for i in range(2,101) if i%2==0]
acc=[]
dic = {}
for i in lis:
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(X_train, y_train)
y_knn_pred = knn.predict(X_test)
  #acc.append(accuracy_score(y_test,y_knn_pred))
  # dic[i] = accuracy_score(y_test,y_knn_pred)
newdf = pd.DataFrame({"Actual":y_test, "Predicted":y_knn_pred})
print(newdf)
print(confusion_matrix(y_test, y_knn_pred))
#print(max(acc))
# 0.8875 = 89%
# 0.875 = 88%

print(dic)