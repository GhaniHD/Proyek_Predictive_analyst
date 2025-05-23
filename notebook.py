men#!/usr/bin/env python
# coding: utf-8

# ## 1.Load Data

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt


# In[7]:


# Load data
df = pd.read_csv("data/Walmart.csv")
print(df.shape)
df.head()


# ## 2. Data Understanding & Removing Outlier

# In[9]:


df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.info()


# In[11]:


# Cek missing value
print(df.isnull().sum())


# In[12]:


# Ringkasan statistik
df.describe()


# In[13]:


# Deteksi outlier pada Weekly_Sales
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Weekly_Sales'])
plt.title("Outlier pada Weekly Sales")
plt.show()


# In[14]:


# Hilangkan outlier ekstrem (lebih dari Q3 + 1.5*IQR)
Q1 = df['Weekly_Sales'].quantile(0.25)
Q3 = df['Weekly_Sales'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df = df[df['Weekly_Sales'] <= upper_bound]


# ## 3. Univariate Analysis

# In[15]:


# Trend penjualan agregat
sales_by_date = df.groupby('Date')['Weekly_Sales'].sum().reset_index()

plt.figure(figsize=(12, 5))
plt.plot(sales_by_date['Date'], sales_by_date['Weekly_Sales'], label='Total Weekly Sales')
plt.title("Weekly Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.show()


# ## 4. Multivariate Analysis

# In[16]:


# Korelasi antara fitur numerik
numerics = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
corr = df[numerics].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# ## 5. Data Preparation

# In[17]:


# Fokus pada total penjualan seluruh toko
sales_df = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
sales_df.columns = ['ds', 'y']
sales_df = sales_df.sort_values('ds')


# In[18]:


# Cek apakah data sudah terurut
sales_df.head()


# ## 6. Modeling

# ### a. Prophet

# In[19]:


prophet_model = Prophet()
prophet_model.fit(sales_df)

# Forecast 12 minggu ke depan
future = prophet_model.make_future_dataframe(periods=12, freq='W')
forecast = prophet_model.predict(future)

# Plot hasil prediksi
prophet_model.plot(forecast)
plt.title("Prophet Forecast")
plt.show()


# ### b.SARIMAX

# In[20]:


# Set index waktu
sarima_df = sales_df.copy()
sarima_df.set_index('ds', inplace=True)

# Split data: train dan test (12 minggu terakhir sebagai test)
train = sarima_df.iloc[:-12]
test = sarima_df.iloc[-12:]

# Fit SARIMAX (param bisa dituning)
sarima_model = SARIMAX(train['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
sarima_result = sarima_model.fit(disp=False)

# Forecast
sarima_forecast = sarima_result.predict(start=test.index[0], end=test.index[-1])

# Plot
plt.figure(figsize=(12, 5))
plt.plot(train.index, train['y'], label='Train')
plt.plot(test.index, test['y'], label='Actual')
plt.plot(test.index, sarima_forecast, label='Forecast')
plt.legend()
plt.title("SARIMA Forecast")
plt.grid(True)
plt.show()


# ## 7. Evaluation

# In[21]:


# Prophet Evaluation
prophet_forecast = forecast.set_index('ds').loc[test.index]['yhat']
mae_prophet = mean_absolute_error(test['y'], prophet_forecast)
rmse_prophet = sqrt(mean_squared_error(test['y'], prophet_forecast))

# SARIMA Evaluation
mae_sarima = mean_absolute_error(test['y'], sarima_forecast)
rmse_sarima = sqrt(mean_squared_error(test['y'], sarima_forecast))

print("=== Model Evaluation ===")
print(f"Prophet MAE: {mae_prophet:.2f}")
print(f"Prophet RMSE: {rmse_prophet:.2f}")
print(f"SARIMA MAE: {mae_sarima:.2f}")
print(f"SARIMA RMSE: {rmse_sarima:.2f}")

