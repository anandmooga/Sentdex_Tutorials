import quandl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

api_key = open('Quandl_api.txt', 'r').read()


housing_data = pd.read_pickle('HPI.pickle')
housing_data = housing_data.pct_change()
print(housing_data.head())
