#!/usr/bin/env python
# coding: utf-8

# ##  Import library

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt


# ## 1.Load Data

# In[24]:


# Load data
df = pd.read_csv("data/Walmart.csv")
print(df.shape)
df.head()


# ## 2. Data Understanding

# In[25]:


# Konversi kolom Date ke format datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
print("\nInfo dataset:")
df.info()


# In[26]:


# Pemeriksaan missing values
print("\nMissing values per kolom:")
print(df.isnull().sum())


# In[27]:


# Analisis statistik deskriptif
print("\nStatistik deskriptif:")
df.describe()


# In[28]:


# Visualisasi distribusi penjualan dan outlier
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['Weekly_Sales'])
plt.title("Boxplot Weekly Sales")

plt.subplot(1, 2, 2)
sns.histplot(df['Weekly_Sales'], kde=True)
plt.title("Distribusi Weekly Sales")
plt.tight_layout()
plt.show()


# ## 3. Data Preparation

# In[29]:


# Penanganan outlier dengan IQR
Q1 = df['Weekly_Sales'].quantile(0.25)
Q3 = df['Weekly_Sales'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR


# In[30]:


print(f"\nBatas atas outlier: {upper_bound:.2f}")
print(f"Jumlah data sebelum pembersihan: {len(df)}")
df = df[df['Weekly_Sales'] <= upper_bound]
print(f"Jumlah data setelah pembersihan: {len(df)}")
print(f"Outlier yang dihapus: {6435-6361} ({((6435-6361)/6435)*100:.2f}%)")


# In[31]:


# Persiapan data untuk modeling
sales_df = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
sales_df.columns = ['ds', 'y']  # Format khusus Prophet
sales_df = sales_df.sort_values('ds')


# In[32]:


print("\nContoh data yang sudah diproses:")
sales_df.head()


# ## 4. Exploratory Data Analysis

# In[33]:


# Analisis tren temporal
plt.figure(figsize=(12, 5))
plt.plot(sales_df['ds'], sales_df['y'])
plt.title("Tren Penjualan Mingguan")
plt.xlabel("Tanggal")
plt.ylabel("Total Penjualan")
plt.grid(True)
plt.show()


# In[34]:


# Analisis korelasi
numerics = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
plt.figure(figsize=(10, 6))
sns.heatmap(df[numerics].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriks Korelasi")
plt.show()


# ## 5. Modeling

# ### 5.1 Prophet Model

# In[35]:


# Inisialisasi dan training model Prophet
prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative'
)

prophet_model.fit(sales_df)


# In[36]:


# Membuat prediksi
future = prophet_model.make_future_dataframe(periods=12, freq='W')
forecast = prophet_model.predict(future)


# In[37]:


# Visualisasi hasil
fig1 = prophet_model.plot(forecast)
plt.title("Forecast dengan Prophet")
plt.show()


# ### 5.2 SARIMAX Model

# In[38]:


# Persiapan data untuk SARIMAX
sarima_df = sales_df.set_index('ds')
train = sarima_df.iloc[:-12]
test = sarima_df.iloc[-12:]


# In[39]:


# Training model SARIMAX
sarima_model = SARIMAX(
    train['y'],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 52),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_result = sarima_model.fit(disp=False)


# In[40]:


# Membuat prediksi dan visualisasi
sarima_forecast = sarima_result.get_forecast(steps=12)
pred_ci = sarima_forecast.conf_int()

plt.figure(figsize=(12, 6))
plt.plot(train.index, train['y'], label='Training')
plt.plot(test.index, test['y'], label='Actual')
plt.plot(test.index, sarima_forecast.predicted_mean, label='Forecast')
plt.fill_between(test.index,
                 pred_ci.iloc[:, 0],
                 pred_ci.iloc[:, 1], color='k', alpha=0.1)
plt.title("SARIMAX Forecast")
plt.legend()
plt.show()


# ## 6. Evaluation

# In[41]:


# Evaluasi Prophet
prophet_forecast = forecast.set_index('ds').loc[test.index]['yhat']
mae_prophet = mean_absolute_error(test['y'], prophet_forecast)
rmse_prophet = sqrt(mean_squared_error(test['y'], prophet_forecast))


# In[42]:


# Evaluasi SARIMAX
sarima_pred = sarima_forecast.predicted_mean
mae_sarima = mean_absolute_error(test['y'], sarima_pred)
rmse_sarima = sqrt(mean_squared_error(test['y'], sarima_pred))


# In[43]:


# Tampilkan hasil evaluasi
print("=== Hasil Evaluasi ===")
print(f"{'Model':<10} | {'MAE':>12} | {'RMSE':>12}")
print("-"*40)
print(f"{'Prophet':<10} | {mae_prophet:12.2f} | {rmse_prophet:12.2f}")
print(f"{'SARIMAX':<10} | {mae_sarima:12.2f} | {rmse_sarima:12.2f}")

