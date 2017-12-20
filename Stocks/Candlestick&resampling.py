import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
style.use('ggplot')

from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates


df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)

df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

##print(df_ohlc.head()) #now as we can see date is the index , bbut candlestick needs ohlc and dates in daes format

df_ohlc.reset_index(inplace=True)
##print(df_ohlc.head()) # and it also needs date in mmdates format

df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

print(df_ohlc.head())

fig = plt.figure()
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1,sharex=ax1)
ax1.xaxis_date() #this displays mdates as normal dates
#candlestick graph ! , green s up , red is down . and its displays ohlc data
candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')

ax2.fill_between(df_volume.index.map(mdates.date2num),df_volume.values,0)

plt.show()
