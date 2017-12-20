import quandl
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

# Not necessary, I just do this so I do not show my API key.
api_key = open('Quandl_api.txt','r').read()

def state_list():
    fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return fiddy_states[0][0][1:]
    

def grab_initial_state_data():
    states = state_list()

    main_df = pd.DataFrame()

    for abbv in states:
        query = "FMAC/HPI_"+str(abbv)
        df = quandl.get(query, authtoken=api_key)
        df.rename(columns={'Value': abbv}, inplace=True)
        df[abbv] = (df[abbv]-df[abbv][0]) / df[abbv][0] * 100.0
        print(df.head())
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)
            
    pickle_out = open('fiddy_states3.pickle','wb')
    pickle.dump(main_df, pickle_out)
    pickle_out.close()

def HPI_Benchmark():
    df = quandl.get("FMAC/HPI_USA", authtoken=api_key)
    df["United States"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    df.rename(columns={'United States':'US_HPI'}, inplace=True)
    return df

def mortgage_30y():
    df = quandl.get("FMAC/MORTG", trim_start="1975-01-01", authtoken=api_key)
    df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    df=df.resample('1D').mean()
    df=df.resample('M').mean()
    return df

def sp500_data():
    df = quandl.get("YAHOO/INDEX_GSPC", trim_start="1975-01-01", authtoken=api_key)
    df["Adjusted Close"] = (df["Adjusted Close"]-df["Adjusted Close"][0]) / df["Adjusted Close"][0] * 100.0
    df=df.resample('M').mean()
    df.rename(columns={'Adjusted Close':'sp500'}, inplace=True)
    df = df['sp500']
    return df

def gdp_data():
    df = quandl.get("BCB/4385", trim_start="1975-01-01", authtoken=api_key)
    df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    df=df.resample('M').mean()
    df.rename(columns={'Value':'GDP'}, inplace=True)
    df = df['GDP']
    return df

def us_unemployment():
    df = quandl.get("ECPI/JOB_G", trim_start="1975-01-01", authtoken=api_key)
    df["Unemployment Rate"] = (df["Unemployment Rate"]-df["Unemployment Rate"][0]) / df["Unemployment Rate"][0] * 100.0
    df=df.resample('1D').mean()
    df=df.resample('M').mean()
    return df



grab_initial_state_data() 
HPI_data = pd.read_pickle('fiddy_states3.pickle')
m30 = mortgage_30y()
sp500 = sp500_data()
gdp = gdp_data()
HPI_Bench = HPI_Benchmark()
unemployment = us_unemployment()
m30.columns=['M30']
HPI = HPI_Bench.join([m30,sp500,gdp,unemployment])
HPI.dropna(inplace=True)
print(HPI.corr())



##import pandas as pd
##import quandl
##import pickle
##import matplotlib.pyplot as plt
##from matplotlib import style
##style.use('fivethirtyeight')
##
##api_key = open('Quandl_api.txt' , 'r').read()
##
##def mortgage_30y():
##    df = quandl.get('FMAC/MORTG', authtoken=api_key)
##    df['Value'] = (df['Value'] - df['Value'][0]) / df['Value'][0] * 100
##    df=df.resample('D')
##    df = df.resample('M',how='mean')
##    df.columns = ['M30']
##    return df 
##
##def state_list():
##    fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
##    return fiddy_states[0][0][1:]
##    
##def grab_initial_state_data():
##    states = state_list()
##
##    main_df = pd.DataFrame()
##
##    for abbv in states:
##        query = "FMAC/HPI_"+str(abbv)
##        df = quandl.get(query, authtoken=api_key)
##        df.rename(columns={'Value':str(abbv)}, inplace=True)
##        #df = df.pct_change()
##        df[abbv] = (df[abbv] - df[abbv][0]) / df[abbv][0] * 100
##        print(query)
##        if main_df.empty:
##            main_df = df
##        else:
##            main_df = main_df.join(df)
##            
##    pickle_out = open('fiddy_states3.pickle','wb')
##    pickle.dump(main_df, pickle_out)
##    pickle_out.close()
##
##def HPI_Benchmark():
##    df = quandl.get('FMAC/HPI_USA', authtoken=api_key)
##    df['Value'] = (df['Value'] - df['Value'][0]) / df['Value'][0] * 100
##   
##    return df
##
##
##def sp500_data():
##    df = quandl.get("YAHOO/INDEX_GSPC", trim_start="1975-01-01", authtoken=api_key)
##    df["Adjusted Close"] = (df["Adjusted Close"]-df["Adjusted Close"][0]) / df["Adjusted Close"][0] * 100.0
##    df=df.resample('M').mean()
##    df.rename(columns={'Adjusted Close':'sp500'}, inplace=True)
##    df = df['sp500']
##    return df
##
##def gdp_data():
##    df = quandl.get("BCB/4385", trim_start="1975-01-01", authtoken=api_key)
##    df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
##    df=df.resample('M').mean()
##    df.rename(columns={'Value':'GDP'}, inplace=True)
##    df = df['GDP']
##    return df
##
##def us_unemployment():
##    df = quandl.get("ECPI/JOB_G", trim_start="1975-01-01", authtoken=api_key)
##    df["Unemployment Rate"] = (df["Unemployment Rate"]-df["Unemployment Rate"][0]) / df["Unemployment Rate"][0] * 100.0
##    df=df.resample('1D').mean()
##    df=df.resample('M').mean()
##    return df
##
##
##
###grab_initial_state_data() 
##HPI_data = pd.read_pickle('fiddy_states3.pickle')
##m30 = mortgage_30y()
##sp500 = sp500_data()
##gdp = gdp_data()
##HPI_Bench = HPI_Benchmark()
##unemployment = us_unemployment()
##m30.columns=['M30']
##HPI = HPI_Bench.join([m30,sp500,gdp,unemployment])
##print(HPI)
##HPI.dropna(inplace=True)
##print(HPI.corr())
##HPI.to_pickle('HPI.pickle')
##    
##
####m30 = mortgage_30y()
####HPI_data = pd.read_pickle('fiddy_states3.pickle')
####HPI_bench= HPI_Benchmark()
####
####state_HPI_M30 = HPI_data.join(m30)
####
####
####print(state_HPI_M30.corr())
##
##
##
##
##
##
##
##
##
####HPI_Benchmark()
##    
###grab_initial_state_data()
####
####fig = plt.figure()
####ax1 = plt.subplot2grid((1,1),(0,0))
##
####fig = plt.figure()
####ax1 = plt.subplot2grid((2,1),(0,0))
####ax2 = plt.subplot2grid((2,1),(1,0), sharex = ax1)
##
##
##
####HPI_data['TX12MA'] = pd.rolling_mean(HPI_data['TX'], 12)
####HPI_data['TX12STD'] = pd.rolling_std(HPI_data['TX'], 12)
####
####print(HPI_data[['TX','TX12MA','TX12STD']].head())
####
#####HPI_data.dropna(how = 'all', inplace=True)
######HPI_data.fillna(value= -99999, inplace=True)
####
####
####HPI_data.dropna(inplace=True)
####HPI_data[['TX','TX12MA']].plot(ax = ax1)
####HPI_data['TX12STD'].plot(ax = ax2)
##
##
####TX_AK_12corr = pd.rolling_corr(HPI_data['TX'], HPI_data['AK'], 12)
####
####HPI_data['TX'].plot(ax=ax1, label='TX HPI')
####HPI_data['AK'].plot(ax=ax1, label='AK HPI')
####
####ax1.legend(loc=4)
####
####TX_AK_12corr.plot(ax=ax2, label= 'TX_AK_12corr')
####
####plt.legend(loc=4)
####plt.show()
##
##
##
##
