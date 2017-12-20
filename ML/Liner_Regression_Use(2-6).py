'''TUT 2 for Liner Regression, the classifier will perform much better if you covert the features into pct changes
and normalise them'''
import pandas as pd
import quandl
import math , datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
style.use('ggplot')


df = quandl.get('WIKI/GOOGL', authtoken="kan1K2QbA6bS31gSsygN")

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close',  'HL_PCT', 'PCT_change', 'Adj. Volume']]

#print(df.head())

'''TUT 3 , identifying the label we did not use Adl. Close becasue HL_PCT and PCT_change are derived using that
and actually in real life we wont have adj.close, so instead we shft the column up to see realtion b/w futer sticl price
and todays features. '''

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
#df.dropna(inplace=True)
#i commented out drop na becasue i want to save the values with nan so that i can predict future on them. tut5

#print(df.tail())

'''TUT 4 , preprocessing is done with all the values so it takes a lot  of time if you do it every tie a new value comes ,
but it helps so choose wisely '''

X = np.array(df.drop(['label'], 1))    #features 
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]


df.dropna(inplace=True)
y = np.array(df['label'])              #label


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

#using liner regression 
#clf = LinearRegression()
#now to increase the speed in train part we do this , ie the number of jos it does at once
clf = LinearRegression(n_jobs=-1) # actally we can put a number like 10 ,  etc. I put -1 because  it means do max you can 

#using svm
#clf = svm.SVR()
#clf = svm.SVR(kernel = 'poly')
#for k in ['linear','poly','rbf','sigmoid']:
#    clf = svm.SVR(kernel=k)
#    clf.fit(X_train, y_train)
#    confidence = clf.score(X_test, y_test)
#    print(k,confidence)

##clf.fit(X_train, y_train)
###used to see how accurate the model is
##'''pickling '''
##with open('linearregression.pickle', 'wb') as f:
##    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)


accuracy = clf.score(X_test, y_test)

#print(accuracy, forecast_out)  # we are printing out % accuracy forecast_out nummber of days in advance !

''' TUT 5, we are predicting future values in this part '''

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set :
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

''' TUT 6, we are going to pickle the model'''
#see code above where pickel is used

