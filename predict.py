#%% [markdown]
# # Predicting the Stock Market
# In this project we use what we learned about machine learning to predict prices in the stock market
# __Disclaimer:__ You should not make trades with any models developed in this mission. The disclaimer on from the website that this guide says as much. The developer who wrote this also has virtually no experience in finance. The material contained in this project does not contain trading advice. Only one's curiosity about how models work.
#%%
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

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
indicators = pd.DataFrame()
intervals = [5, 30, 365]
base_features = ['Close', 'Volume']

def generate_features(df, interval, measures):
    for measure in measures:
        for interval in intervals:
            indicators[f'avg_{measure.lower()}_last_{interval}_day'] = df[measure].rolling(f'{interval}D').mean().shift(axis=0, periods=1)
            indicators[f'stdev_{measure.lower()}_last_{interval}_day'] = df[measure].rolling(f'{interval}D').std().shift(axis=0, periods=1)

generate_features(stocks, intervals, base_features)

indicators['avg_ratio_close_5_365_day'] = indicators['avg_close_last_5_day'] / indicators['avg_close_last_365_day']
indicators['avg_ratio_volume_5_365_day'] = indicators['avg_volume_last_5_day'] / indicators['avg_volume_last_365_day']
print(indicators.columns)

#%% [markdown]
# Next, we'll take a look at the ratio between the low and high prices in the past year compared to the current price

#%%
year_min_max = pd.DataFrame()
year_min_max['min'] = stocks.groupby(pd.Grouper(freq='Y'))['Close'].min()
year_min_max['max'] = stocks.groupby(pd.Grouper(freq='Y'))['Close'].max()
year_min_max['year'] = year_min_max.index.year
year_min_max.index = year_min_max['year']
year_min_max = year_min_max.drop('year', axis=1)
indicators[indicators.index.year > 1949]['prev_year_close_min'] = year_min_max['min']

#%%
indicators.columns

#%%
indicators.head()


#%%
stocks = pd.merge(stocks, indicators, left_on='Date', right_on='Date')

#%%
stocks.head()

#%% [markdown]
# While pandas is still able to calculated a rolling window if there isn't enough days, this isn't good enough for what we need. We have to drop columns that are older than a year old.

#%%
stocks = stocks[stocks.index > datetime(year=1951, month=1, day=2)]
stocks = stocks.dropna(axis=0)

#%% [markdown]
#Because we are making predictions using a time series dataset, we'll want to split our train and test date based on the data

#%%
train = stocks[stocks.index < datetime(year=2013, month=1, day=1)]
test = stocks[stocks.index >= datetime(year=2013, month=1, day=1)]


#%% [markdown]
# ###Running the Model
# Before training our model, we'll want to define something that we can evaluate our model's performance on.
# _Mean Squared Error_ is a common metric. However, because it squares the error, it is harder for us to tell intuitively how far we are from the true price.
# In this project, the metric that will be used is _Mean Absolute Error_. This metric may be more suited for time series data, and it is also more intuitive in telling us how far off from the true price we are. 

#%%

X = ['avg_last_5_day', 'avg_last_30_day', 'avg_last_365_day', 'avg_ratio_5_365_day', 'stdev_last_5_day', 'stdev_last_30_day', 'stdev_last_365_day']
y = ['Close']
reggie = LinearRegression().fit(train[X], train[y])
predictions = reggie.predict(test[X])
mae = mean_absolute_error(test[y], predictions)

#%%
print(mae)

#%% [markdown]
# The model gives us an MAE of about $14.32. Can we reduce the error further?
# Let's see if we can engineer additional features.
