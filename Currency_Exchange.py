# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 06:29:28 2019

@author: khushal
"""

### https://forex-python.readthedocs.io/en/latest/usage.html

import time
start_time = time.time()

from forex_python.converter import CurrencyRates
from forex_python.converter import CurrencyCodes
import datetime
import pandas as pd
import numpy as np

''' 
Packages for embedding graph on webpage
'''
from bokeh.plotting import figure, output_file ,show
from bokeh.models import ColumnDataSource,DatetimeTickFormatter
from bokeh.models import HoverTool, WheelZoomTool,ResetTool,SaveTool


'''
Packages for applying time series LSTM model
'''
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM,Masking

import ctypes  # An included library with Python install.
import win32gui, win32con
import os
# fix random seed for reproducibility
np.random.seed(7)

#Start_date = datetime.datetime.now() + datetime.timedelta(-120)
Start_date = datetime.datetime.now() + datetime.timedelta(-122)
Current_date = datetime.datetime.now() + datetime.timedelta(-2)
#Current_date = datetime.datetime.now()
print("Start_date = ",Start_date,"\n Current_date = ",Current_date)

def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

def extract_historical_currency_rates(Start_date,Current_date):
    date_value_df = pd.date_range(start=Start_date, end=Current_date)
    #print(date_value_df)
    
    currency_df = pd.DataFrame(columns=('Date','USD_to_INR_Rate','GBP_to_INR_Rate'))
                                        #,'INR_symbol'))
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
        #print("USD_to_INR_Rate = ",USD_to_INR_Rate)
        #print("GBP_to_INR_Rate = ", GBP_to_INR_Rate)
    
    c = CurrencyCodes()
    #currency_df['INR_symbol'] =  c.get_symbol('INR')
    currency_df['Date'] = currency_df['Date'].dt.strftime('%m/%d/%Y')
    currency_df['Date'] = pd.to_datetime(currency_df.Date,format='%m/%d/%Y')
    #print("symbol = ",currency_df['INR_symbol'])
    currency_df.index= currency_df['Date']
    # currency_df.drop(['Date'], axis=1,inplace=True) # activate this line to do Graph plotting
    print(currency_df)
    return currency_df

def generate_trend_analysis_on_historical_data(currency_df):
    N = 10
    x_min = currency_df['Date'].tail(30).min() - pd.Timedelta(days=0.1*N)
    x_max = currency_df['Date'].tail(30).max()
    output_file('USD_to_INR_Rate.html',
                    title='USD_to_INR_Rate')    
    source_hover_10 = ColumnDataSource(data=dict(Date=currency_df['Date'].tail(30).tolist(),
                                                  USD_to_INR_Rate=currency_df['USD_to_INR_Rate'].head(30).tolist()
                                                  ))
    
    # show the tooltip
    hover_10 = HoverTool(tooltips=[
                ("% USD_to_INR_Rate", "@USD_to_INR_Rate"),
                ])

    l_10 = figure(title="USD_to_INR_Rate Trend Analysis",
               x_range = (x_min, x_max),width=800,height=500,
               #logo=None,
               x_axis_type="datetime",tools=[hover_10])
    l_10.circle('Date','USD_to_INR_Rate', size=3, color='red',source=source_hover_10,legend='% USD_to_INR_Rate')
    l_10.line('Date','USD_to_INR_Rate',source=source_hover_10, legend='% USD_to_INR_Rate', color='red')    
    l_10.add_tools(ResetTool(),SaveTool(),WheelZoomTool())
    l_10.legend.location = "top_left"
    l_10.title_location = "above"
    l_10.toolbar.logo = None
    l_10.legend.click_policy = "hide"
    l_10.sizing_mode = "stretch_both"
    l_10.xaxis.formatter=DatetimeTickFormatter(
                    days = ['%m/%d', '%a%d'],
                    months = ['%m/%Y', '%b %Y']
                    )
    show(l_10)

# Get Historical Currency data
currency_df = extract_historical_currency_rates(Start_date,Current_date)  

data = currency_df.sort_index(ascending=True,axis=0)
new_data = pd.DataFrame(index=range(0,len(currency_df)),columns=['Date','USD_to_INR_Rate'])

for i in range(len(currency_df)):
    new_data['Date'][i] = currency_df['Date'][i]
    new_data['USD_to_INR_Rate'][i] = currency_df['USD_to_INR_Rate'][i]
    #new_data['GBP_to_INR_Rate'][i] = currency_df['GBP_to_INR_Rate'][i]

print("new_data shape= ",new_data.shape)

new_data.index = new_data.Date
new_data.drop(['Date'], axis=1,inplace=True) 

dataset = new_data.values

train = dataset[0:108,:]
valid = dataset[108:,:]

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [],[]

for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train= np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

print("len(train)",len(train))
print("x_train, y_train ",len(x_train), len(y_train ))

model= Sequential()

# Masking layer for pre-trained embeddings
model.add(Masking(mask_value=0.0))

# Recurrent layer
model.add(LSTM(units=5,return_sequences=True,activation='relu',
               input_shape=(np.array(x_train).shape[1],1)))

model.add(Dropout(0.2))

#model.add(LSTM(units=50, return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(units=50, return_sequences=True))
#model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(x_train,y_train,epochs=70,batch_size=48,verbose=2)
#model.fit(x_train,y_train,epochs=335,batch_size=80,verbose=2) ##Very Nearer Prediction Values

inputs = new_data[len(new_data)-len(valid)-60:].values
inputs= inputs.reshape(-1,1)
print("inputs = ",len(inputs)) 
inputs= scaler.transform(inputs)
 
x_test = []

for i in range(60,inputs.shape[0]):
    x_test.append(inputs[i-60:i,0])

x_test = np.array(x_test)

print("x_test len =",len(x_test))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

price = model.predict(x_test)#,verbose=1)
price = scaler.inverse_transform(price)

rms = np.sqrt(np.mean(np.power((valid-price),2)))

print('Train Score: %.2f RMSE' % (rms))

print("Len of price",len(price))

for i in range(3):
    price_value = np.round(price[i],2)
    Future_date = Current_date + datetime.timedelta(i+1)
    print("price = ",price_value," Future Date of ", Future_date)


def  display_predicted_value_on_msgbox(price):
    i = 0 # for next day, incremeting the value of i till len(price) to get prediction for those many days
    price_value = np.round(price[i],2)
    Future_date = Current_date + datetime.timedelta(i+1)
    str_msg0 = "For " + str(Future_date.strftime("%d %b %Y ")) + ",Predicted USD-INR value is "+ str(price_value)
    i = 2 # for next day, incremeting the value of i till len(price) to get prediction for those many days
    price_value = np.round(price[i],2)
    Future_date = Current_date + datetime.timedelta(i+1)
    str_msg1 = "For " + str(Future_date.strftime("%d %b %Y ")) + ",Predicted USD-INR value is "+ str(price_value)
    str_msg = str_msg0 + "\n"+ str_msg1
    Mbox('Predicted USD-INR value ', str_msg , 1)    
# Generate Graphical trend chart for analysis using Historical Currency data
#generate_trend_analysis_on_historical_data(currency_df)  

display_predicted_value_on_msgbox(price)

def convert(n): 
	return str(datetime.timedelta(seconds = n)) 
	
n =  time.time() - start_time

def convert_sec(n): 
    return str(datetime.timedelta(seconds = n))

print("---Execution Time ---",convert_sec(n))