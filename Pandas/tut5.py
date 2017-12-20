import pandas as pd
import quandl
import pickle

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
        print(query)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)
            
    pickle_out = open('fiddy_states.pickle','wb')
    pickle.dump(main_df, pickle_out)
    pickle_out.close()

#grab_initial_state_data()

pickle_in = open('fiddy_states.pickle' , 'rb')
HPI_data = pickle.load(pickle_in)
print(HPI_data)
pickle_in.close()
#same thing usind pandas
HPI_data.to_pickle('pickle.pickle')
HPI_data2 = pd.read_pickle('pickle.pickle')
print(HPI_data2)





##fifty_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
##abb = fifty_states[0][0]
##
##print (abb)
##for abv in abb[1:] :
##    query= 'FMAC/HPI_'+str(abv)
##    df = quandl.get(query, authtoken=api_key)
##    df.rename(columns={'Value':str(abv)}, inplace=True)
##    if main_df.empty:
##        main_df = df
##    else: 
##        main_df = main_df.join(df)
##        
##print(main_df.head())
####main_df.to_csv('hpia.csv')

