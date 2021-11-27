# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:09:24 2020

@author: srdug
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn import metrics


wineData = pd.read_csv('CMP6202_lab1_wine.csv')
scaler = MinMaxScaler()
standard_scaler = StandardScaler()
label_encoder = LabelEncoder()
wineData["type"] = label_encoder.fit_transform(wineData["type"])
x = wineData.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,13]]
y = wineData.iloc[:, 14]
x = pd.DataFrame(standard_scaler.fit_transform(x, {'copy':False}), 
                            index=x.index, 
                            columns=x.columns)
x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=0)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
tree = RandomForestClassifier()
tree = KNeighborsClassifier(n_neighbors=10)
tree.fit(x_train, y_train)
y_prediction = tree.predict(x_test)
print(y_prediction)

print ("Accuracy:", metrics.accuracy_score(y_test, y_prediction))