#http://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption#
#im going to try clusering on this data set, to predict total power consumed slab or power, well see which goes better; total power = sqrt(activepower**2 + reactivepower**2) 
#this is a rather easy set as data is completly numerical

import pandas as pd
import numpy as np
from sklearn import  preprocessing, cross_validation, cluster



df = pd.read_csv('household_power_consumption.txt', sep = ';')

##print(df.head())
##print(df.info())

df.drop(['Date', 'Time'], 1, inplace =True)

df.dropna(how = 'any', inplace = True)
#data loss is negligiblle 

#we have a problem that as all the values are stored as strings we cannot operate with them, so change them 
df =df.astype(float)
df['Global_power'] = ((df['Global_active_power']**2) + (df['Global_reactive_power']**2))**0.5

##print(df.head())
##print(df.describe())

#making slabs
df['Power_band'] = pd.cut(df['Global_power'], 5)
print(df[['Power_band','Global_power']].groupby(['Power_band'], as_index = False).mean())

## now ill be assiging slabs
df.loc[df['Global_power'] <= 2.285 , 'Global_power'] = 0
df.loc[(df['Global_power'] > 2.285) & (df['Global_power'] <= 4.495) , 'Global_power'] = 1
df.loc[(df['Global_power'] > 4.495) & (df['Global_power'] <= 6.704) , 'Global_power'] = 2
df.loc[(df['Global_power'] > 6.704) & (df['Global_power'] <= 8.914) , 'Global_power'] = 3
df.loc[(df['Global_power'] > 8.914) , 'Global_power'] = 4

#now we have a huge bias , if we input any of the power as input we will get a very high accraccy since we are feeding the answer directy
#so ill be deleting biased columns , you can see the correlation for refrence 
df.drop(['Power_band'], 1, inplace =True)
print(df.head())
##print(df.corr())

#ill be ddeleting some columsn that are useless, and one of the meters so that i can predict the total power without having 3 meters, ill the removing the one with water heater and ac, as they have max cnsumption 
df.drop(['Global_active_power', 'Global_reactive_power', 'Sub_metering_3'], 1, inplace =True)

print(df.head())

#applying out clustering algo

X = np.array(df.drop(['Global_power'], 1))
X = preprocessing.scale(X)
y = np.array(df['Global_power'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
# be careful before you run it, you may run out of memory
clf = cluster.KMeans(n_clusters= 5)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)


