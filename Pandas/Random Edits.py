import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

df = pd.read_csv('F:\ZILL-Z77006_3B.csv')
df= df.set_index('Date')
print(df.head())

df.columns = ['Austin_HPI']
print(df.head())

df.to_csv('newcsv.csv', header = False)

df = pd.read_csv('newcsv.csv', names = ['date','hpi'], index_col =0 )
print(df.head())

df.to_html('sample.html')



##web_stats = {'Day': [1,2,3,4,5,6],
##             'Visitors' : [43,53,34,45,64,34],
##             'Bounce_Rate': [65,72,62,64,54,66]
##             }
##
##
##df= pd.DataFrame(web_stats)
##
###print(df)
###print(df.head())
###print(df.tail(1))
####
####df = df.set_index('Day')
####
####print(df['Visitors'])
####
####
##
##print(df['Visitors'].tolist())
##print(np.array(df[['Bounce_Rate', 'Visitors']]))


