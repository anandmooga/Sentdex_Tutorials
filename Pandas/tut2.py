import pandas as pd
import quandl as qd

##api_key = open('Quandl_api.txt' , 'r').read()
##
##df = qd.get("FMAC/HPI_AK", authtoken=api_key)
##
##print(df.head())

fifty_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
#print(fifty_states[0][0])

abb = fifty_states[0][0]
abb.pop(0)
for abv in abb :
    print('FMAC/HPI_'+str(abv))
