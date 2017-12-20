#https://archive.ics.uci.edu/ml/datasets/Air+quality
#Data set on air pollution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib  import style
from sklearn import preprocessing, cross_validation, linear_model, svm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
style.use('fivethirtyeight')

df = pd.read_excel('AirQualityUCI.xlsx')
df.info()
df.drop(['Date', 'Time', 'NMHC(GT)'], 1, inplace=True)
'''conveting incoorect reading to NaN'''
df.replace(-200, np.NaN, inplace=True)     

##df.dropna(how = 'any', inplace= True)
'''dropiing entries which have too much missing data'''
df. dropna(thresh = 9, inplace = True)

'''filling na values which will not compromise the integrity of the dataset'''
df.fillna(method='bfill', inplace = True)
##df = df[pd.notnull(df['CO(GT)'])]

'''converting the values to pct  change from the beginnig to normalizethe values '''
columns = list(df.columns.values)
for column in columns:
    df[column]= (df[column] - df[column][0]) / df[column][0] * 100

##df.to_excel('AirQualityUCI_processed.xlsx', sheet_name='Sheet1')

##df.plot()
##plt.legend().remove()
##plt.show()

X = np.array(df.drop(['CO(GT)'], 1))
X = preprocessing.scale(X)
y = np.array(df['CO(GT)'])

##print(len(X), len(y))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
''' regression:https://www.analyticsvidhya.com/blog/2015/08/comprehensive-guide-regression/
we will be using various models of regression and seeing which suits the data. '''
##clf =  linear_model.LinearRegression()  #79% avg
##clf = linear_model.Ridge(alpha=0.5)  # 78% #ridge is used for multicolineraity , in thidata we do not have multicolineraity
##clf = linear_model.Lasso(alpha = 0.5)  # 78% same as ridge except in penlity we use w directlt therefore sign in considered, in ridge its w**2
##clf = make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression()) # 81%

##clf.fit(X_train, y_train)
##accuracy = clf.score(X_test, y_test)


avg = 0
for i in range(10):
    clf = make_pipeline(PolynomialFeatures(3), linear_model.LinearRegression()) #87%
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    avg += accuracy
print(avg/10) 

'''applying svm'''

##for k in ['linear','poly','rbf','sigmoid']: #linear  kernel is the best, 
##    clf = svm.SVR(kernel=k) 
##    clf.fit(X_train, y_train)
##    confidence = clf.score(X_test, y_test)
##    print(k,confidence)
    
'''polynomial regresssion of the order 3 with linear regression is the best  fit for the data  '''



