# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 01:22:23 2020

@author: Sriharsha Komera
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import pickle




dataset=pd.read_csv("F:\\Coursera\\Simple Linear Regression\\FuelConsumptionCo2.csv")
dataset.head()
dataset.describe()

le=LabelEncoder()
dataset['MAKE']=le.fit_transform(dataset['MAKE'])
dataset['MODEL']=le.fit_transform(dataset['MODEL'])
dataset['VEHICLECLASS']=le.fit_transform(dataset['VEHICLECLASS'])
dataset['TRANSMISSION']=le.fit_transform(dataset['TRANSMISSION'])
dataset['FUELTYPE']=le.fit_transform(dataset['FUELTYPE'])

X=dataset.iloc[:,2:11]
y=dataset.iloc[:,-1]



bestfeatures=SelectKBest(score_func=chi2,k=8)
fit=bestfeatures.fit(X,y)

dfscores=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(X.columns)

featurescores=pd.concat([dfcolumns,dfscores],axis=1)
featurescores.columns=['Specs','Score']

featurescores

print(featurescores.nlargest(10,'Score'))
model=ExtraTreesClassifier()
model.fit(X,y)

print(model.feature_importances_)
feat_importance=pd.Series(model.feature_importances_,index=X.columns)
feat_importance.nlargest(10).plot(kind='barh')
plt.show()

corrmat=dataset.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,30))
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True, cmap='RdYlGn')

X=X[["CYLINDERS","ENGINESIZE","FUELCONSUMPTION_HWY","FUELCONSUMPTION_COMB","FUELCONSUMPTION_CITY"]]
y=dataset.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

LR=LinearRegression()
LR.fit(X_train,y_train)

lasso=Lasso()
print(cross_val_score(lasso,X,y,cv=10).mean())

y_pred=LR.predict(X_test)
r2_score(y_pred,y_test)

pickle.dump(LR,open('model_Fuel.pkl','wb'))

model_Fuel=pickle.load(open('model_Fuel.pkl','rb'))

print(model_Fuel.predict([[6,3.5,12.1,8.6,10.6]]))