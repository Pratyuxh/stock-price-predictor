
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df= pd.read_csv('CUR-INR.csv')
X = df.iloc[:,:-1].values 
y = df.iloc[:, 1].values

import datetime as dt
df['DATE'] = pd.to_datetime(df['DATE'])
df['DATE']= df['DATE'].map(dt.datetime.toordinal)
df = df.set_index('DATE', append=False)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

"""from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
X_train = enc.fit_transform(X_train)
X_test = enc.transform(X_test)
enc.categories"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
