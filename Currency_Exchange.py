# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 06:29:28 2019

@author: khushal
"""

### https://forex-python.readthedocs.io/en/latest/usage.html

from forex_python.converter import CurrencyRates
from forex_python.converter import CurrencyCodes
import datetime
import pandas as pd

Start_date = datetime.datetime.now() + datetime.timedelta(-30)
Current_date = datetime.datetime.now()
print(Start_date,Current_date)

date_value_df = pd.date_range(start=Start_date, end=Current_date)
#print(date_value_df)

currency_df = pd.DataFrame(columns=('Date','USD_to_INR_Rate','GBP_to_INR_Rate',
                                    'INR_symbol'))
USD_to_INR_Rate = []
GBP_to_INR_Rate = []

#print(currency_df)
for i in range(len(date_value_df)):
    c = CurrencyRates()
    #print(date_value_df[i])
    USD_to_INR_Rate = round(c.get_rates('USD',date_value_df[i]).get('INR'),2)
    GBP_to_INR_Rate = round(c.get_rates('GBP',date_value_df[i]).get('INR'),2)
    dict_rates = {'Date':date_value_df[i],'USD_to_INR_Rate':USD_to_INR_Rate,'GBP_to_INR_Rate':GBP_to_INR_Rate}
    currency_df = currency_df.append(dict_rates,ignore_index=True)
    print("USD_to_INR_Rate = ",USD_to_INR_Rate)
    print("GBP_to_INR_Rate = ", GBP_to_INR_Rate)

c = CurrencyCodes()
currency_df['INR_symbol'] =  c.get_symbol('INR')
currency_df['Date'] = currency_df['Date'].dt.strftime('%m/%d/%Y')
#print("symbol = ",currency_df['INR_symbol'])

print(currency_df)


