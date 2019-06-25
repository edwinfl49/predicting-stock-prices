#%% [markdown]
# # Predicting the Stock Market
# In this project, we will build a linear regression model to predict the index price of the S&P 500.
#
# __Disclaimer:__ You should not make trades with any models developed in this mission. The disclaimer on from the website that this guide says as much. The developer who wrote this also has virtually no experience in finance. The material contained in this project does not contain trading advice. Only one's curiosity about how models work.
#%%
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import seaborn as sns

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
indicators['year'] = indicators.index.year

num_rows_first_year = indicators[indicators['year'] == 1950].shape[0]
indicators['prev_year_close_min'] = indicators['year'].apply(lambda y: year_min_max.loc[y, 'min']).shift(axis=0, periods=num_rows_first_year)
indicators['prev_year_close_max'] = indicators['year'].apply(lambda y: year_min_max.loc[y, 'max']).shift(axis=0, periods=num_rows_first_year)

indicators['close'] = stocks['Close']
indicators['ratio_prev_year_low_to_current_close'] = indicators['prev_year_close_min'] / indicators['close']
indicators['ratio_prev_year_high_to_current_close'] = indicators['prev_year_close_max'] / indicators['close'] 
indicators = indicators.drop(['close'], axis=1)
#%% [markdown]
# Going off on conventional knowledge, the value of the stock market in general has only increased, meaning we can expect that the year would have a large effect on the price. 
# Instead of using the year as it is presented, we'll subtract all year values by 1950 so that 1950 will be year 0 and 2000 will be 50.
#%% 
indicators['year'] = indicators['year'] - 1950
#%%
indicators.describe()
#%%
stocks = pd.merge(stocks, indicators, left_on='Date', right_on='Date')
#%%
stocks.head()
#%% [markdown]
# While pandas is still able to calculated a rolling window if there isn't enough days, this isn't good enough for what we need. We have to drop columns that are older than a year old.
# We'll also drop columns with NA values, as well as any columns we used primarily to create other columns.
#%%
stocks = stocks[stocks.index > datetime(year=1951, month=1, day=2)]
stocks = stocks.drop(['prev_year_close_min', 'prev_year_close_max'], axis=1)
stocks = stocks.dropna(axis=0)
#%% [markdown]
# With our features in line, let's display some of the data graphically to get an idea of how they interact with the closing price
#%%
stocks.plot.line(y='Close', use_index=True)
#%%
sns.heatmap(stocks.drop(['Open', 'High', 'Low', 'Adj Close'], axis=1).corr().abs())
#%% [markdown]
# ### Running the Model
# Before training our model, we'll want to define something that we can evaluate our model's performance on.
# _Mean Squared Error_ is a common metric. However, because it squares the error, it is harder for us to tell intuitively how far we are from the true price.
# In this project, the metric that will be used is _Mean Absolute Error_. This metric may be more suited for time series data, and it is also more intuitive in telling us how far off from the true price we are. 
#%% [markdown]
# As we examined, the avg ratios 5-365 days and ratio between high/low and current price appear to have very little correlation, so we'll drop those...
#%%
# last index of columns in original data is 5
X = stocks.columns[6:].to_list()
X = [c for c in X if c not in ('avg_ratio_close_5_365_day', 'avg_ratio_volume_5_365_day', 'ratio_prev_year_low_to_current_close', 'ratio_prev_year_high_to_current_close')]
print(X)
#%% [markdown]
#Because we are making predictions using a time series dataset, we'll want to split our train and test date based on the data
#%%
train = stocks[stocks.index < datetime(year=2013, month=1, day=1)]
test = stocks[stocks.index >= datetime(year=2013, month=1, day=1)]
#%%
y = ['Close']
reggie = LinearRegression().fit(train[X], train[y])
predictions = reggie.predict(test[X])
mae = mean_absolute_error(test[y], predictions)
r2 = r2_score(test[y], predictions)
#%%
print("Mean Absolute Error:", mae)
print("Variance Score:", r2)
#%%
# ### Plotting Actual vs Predictions
plt.scatter(test[y], predictions)
#%%
# ### Plotting Predictions
test_with_predictions = test.copy()
test_with_predictions['predicted'] = predictions

plt.plot(train.index, train['Close'], label="Historical Close")
plt.plot(test.index, test['Close'], label="True Close (2013+)")
plt.scatter(test_with_predictions.index, test_with_predictions['predicted'], label="Predicted Close (2013+)")
plt.legend()
#%% [markdown]
# ### Conclusion
# The model gives us an MAE of about $14.36 and an R2 score that is very close to zero. While the numbers look great, the fact that our error is this immediately raises some flags. 
# - Did we overfit the model?
# - Is this sample a random walk?
# Further evaluation of the model and data is required.
# 
# As for improving the model itself, some ideas include:
# - Add more features, including
#   - Date components (day, week, month, holidays in the previous month)
#   - Other features not mentoined
#   - Incorporate outside data into this analysis
# - Make predictions only one day head
#   - To do this we would train the model using 1951-01-03 to 2013-01-02 to make predictions for 1951-01-03
#   - Then we add the past day's data (2013-01-03) and use it to predict data for January 4th, and so on
