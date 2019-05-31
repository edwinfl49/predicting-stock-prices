#%% [markdown]
# # Predicting the Stock Market
# In this project we use what we learned about machine learning to predict prices in the stock market
# __Disclaimer:__ You should not make trades with any models developed in this mission. The disclaimer on from the website that this guide says as much. The developer who wrote this also has virtually no experience in finance. The material contained in this project does not contain trading advice. Only one's curiosity about how models work.
#%%
import pandas as pd
from datetime import datetime

stocks = pd.read_csv('data/sphist.csv')
stocks['Date'] = pd.to_datetime(stocks['Date'])
stocks = stocks.set_index('Date').sort_values('Date')
stocks.head(10)

#%% [markdown]
# ### An Important Note About Stock Market Data
# Stock market data is sequential (ie time series), and each observation of a particular day comes after the previous day, meaning that each row is not an independent observation.
# This means it is easier to accidentally inject _future_ knowledge into past observations if we're not careful.
# Several of the features/indicators we'll create involve something to the degree of average price of the last _n_ days. When computing these, we have to becareful that the current day is not included in this measure.
# For example, if we're computing the average price of the last 5 days from 1995-09-25, then we need to make sure that 1995-09-25 is not in included in the calculation.

#%%
stocks['Close'].rolling('5D').mean()

#%%
stocks['Close'].rolling('5D').mean().shift(axis=0, periods=1)


#%%
