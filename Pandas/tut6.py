import pandas as pd
import quandl
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

api_key = open('Quandl_api.txt' , 'r').read()


def state_list():
    fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return fiddy_states[0][0][1:]
    

def grab_initial_state_data():
    states = state_list()

    main_df = pd.DataFrame()

    for abbv in states:
        query = "FMAC/HPI_"+str(abbv)
        df = quandl.get(query, authtoken=api_key)
        df.rename(columns={'Value':str(abbv)}, inplace=True)
        #df = df.pct_change()
        df[abbv] = (df[abbv] - df[abbv][0]) / df[abbv][0] * 100
        print(query)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)
            
    pickle_out = open('fiddy_states3.pickle','wb')
    pickle.dump(main_df, pickle_out)
    pickle_out.close()

def HPI_Benchmark():
    df = quandl.get('FMAC/HPI_USA', authtoken=api_key)
    df['Value'] = (df['Value'] - df['Value'][0]) / df['Value'][0] * 100
    return df

HPI_Benchmark()
    
#grab_initial_state_data()
##
##fig = plt.figure()
##ax1 = plt.subplot2grid((1,1),(0,0))

HPI_data = pd.read_pickle('fiddy_states3.pickle')
##benchmark = HPI_Benchmark()
##HPI_data.plot(ax = ax1)
##benchmark.plot(ax = ax1, color= 'k', linewidth = 10)
##
##
##plt.legend().remove()
##plt.show()

HPI_State_Correlation = HPI_data.corr()
print(HPI_State_Correlation)

print(HPI_State_Correlation.describe())

