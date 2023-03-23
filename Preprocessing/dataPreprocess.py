import pandas as pd
import os
from datetime import datetime


def convert_currency(var):
    """
    convert the string number to a float
    """
    new_value = var.replace(",", "").replace("$", "")
    if var.find("K"):
        new_value = var.replace("K", "").replace(",", "")
        new_value = float(new_value) * 1000
    return float(new_value)


df_bc = pd.read_csv('raw_data/bitcoin_raw.csv')
df_gold = pd.read_csv('raw_data/gold_df.csv')
df_oil = pd.read_csv('raw_data/oil_price_raw.csv')
df_raw = pd.read_csv('raw_data/raw_v2_dataset.csv')
df_vix = pd.read_csv('raw_data/vixcurrent.csv', skiprows=[0])

df_bc['Date'] = pd.to_datetime(df_bc.Date)
df_gold['Date'] = pd.to_datetime(df_gold.Date)
df_oil['Date'] = pd.to_datetime(df_oil.Date)
df_raw['Date'] = pd.to_datetime(df_raw.Date)
df_vix['Date'] = pd.to_datetime(df_vix.Date)

df_bc = df_bc.sort_values('Date')
df_gold = df_gold.sort_values('Date')
df_oil = df_oil.sort_values('Date')
df_raw = df_raw.sort_values('Date')
df_vix = df_vix.sort_values('Date')

df_bc.set_index('Date', inplace=True, drop=False)
df_gold.set_index('Date', inplace=True, drop=True)
df_oil.set_index('Date', inplace=True, drop=True)
df_raw.set_index('Date', inplace=True, drop=True)
df_vix.set_index('Date', inplace=True, drop=True)

df_bc.drop(['Currency'], axis=1, inplace=True)
df_bc.rename({'Closing Price (USD)': 'BTC_Close', '24h Open (USD)': 'BTC_Open',
             '24h High (USD)': 'BTC_High', '24h Low (USD)': 'BTC_Low'}, axis='columns', inplace=True)
df_gold.rename({'Price': 'gold_Close', 'Open': 'gold_Open', 'High': 'gold_High', 'Low': 'gold_Low',
               'Volume': 'gold_Volume', 'Chg%': 'gold_Chg%'}, axis='columns', inplace=True)
df_gold.drop(['gold_Close'], axis=1, inplace=True)
df_gold['gold_Volume'] = df_gold['gold_Volume'].str.strip('K').astype(float)*1000
df_gold['gold_Open'] = df_gold['gold_Open'].str.replace(',','').astype(float)
df_gold['gold_High'] = df_gold['gold_High'].str.replace(',','').astype(float)
df_gold['gold_Low'] = df_gold['gold_Low'].str.replace(',','').astype(float)

df_gold['gold_Chg%'] = df_gold['gold_Chg%'].str.strip('%').astype(float)/100
df_oil.rename({'Close': 'oil_Close', 'Open': 'oil_Open', 'High': 'oil_High', 'Low': 'oil_Low',
              'Volume': 'oil_Volume', 'Adj Close': 'oil_Adj Close'}, axis='columns', inplace=True)
df_oil.drop(['oil_Adj Close'], axis=1, inplace=True)
df_raw.drop(df_raw.columns[[0]], axis=1, inplace=True)
df_raw.rename({'BTC price [USD]': 'BTC_Close', 'Volume BTC': 'BTC_Volume', 'Gold price[USD]': 'gold_Close', 'Oil WTI price[USD]': 'oil_WTI price', 'SP500 close index': 'SP500_Close index',
              'BTC n-transactions': 'BTC_transactions', 'BTC google search interest': 'BTC_interesst', 'VIX Close': 'VIX_Close'}, axis='columns', inplace=True)
df_raw.drop(['BTC_Close', 'oil_WTI price', 'VIX_Close'], axis=1, inplace=True)
df_vix.rename({'VIX Close': 'VIX_Close', 'VIX Open': 'VIX_Open',
              'VIX High': 'VIX_High', 'VIX Low': 'VIX_Low'}, axis='columns', inplace=True)


# df_gold = df_gold.add_prefix("gold_")
df = pd.concat([df_bc, df_gold, df_oil, df_raw, df_vix], axis=1, join='inner')
df.dropna(axis=0, inplace=True)
orders = ['Date', 'BTC_Close', 'BTC_Open', 'BTC_High', 'BTC_Low', 'BTC_Volume', 'BTC_transactions', 'BTC_interesst', 'gold_Open', 'gold_Close', 'gold_High', 'gold_Low',
          'gold_Volume', 'gold_Chg%', 'oil_Open', 'oil_Close', 'oil_High', 'oil_Low', 'oil_Volume', 'VIX_Open', 'VIX_Close', 'VIX_High', 'VIX_Low', 'SP500_Close index']
df = df[orders]

print(df.info())

df.to_csv(os.path.dirname(
    __file__)+"/train_data.csv", index=False)
