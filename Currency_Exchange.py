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

#Start_date = datetime.datetime.now() + datetime.timedelta(-120)
Start_date = datetime.datetime.now() + datetime.timedelta(-602)
Current_date = datetime.datetime.now() + datetime.timedelta(-2)
#Current_date = datetime.datetime.now()
print("Start_date = ",Start_date,"\n Current_date = ",Current_date)

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

'''
import matplotlib.pyplot as plt

#%matplotlib inline

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,15
plt.figure(figsize=(16,8))
plt.plot(currency_df,label='Currency Exchange w.r.t Time')
'''
##################

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM,Masking

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

train = dataset[0:498,:]
valid = dataset[498:,:]

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [],[]

for i in range(15,len(train)):
    x_train.append(scaled_data[i-15:i,0])
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
'''
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
'''
model.add(LSTM(units = 50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(x_train,y_train,epochs=25,batch_size=80,verbose=2)
#model.fit(x_train,y_train,epochs=335,batch_size=80,verbose=2) ##Very Nearer Prediction Values

inputs = new_data[len(new_data)-len(valid)-15:].values
inputs= inputs.reshape(-1,1)
print("inputs = ",len(inputs)) 
inputs= scaler.transform(inputs)
 
x_test = []

for i in range(15,inputs.shape[0]):
    x_test.append(inputs[i-15:i,0])

x_test = np.array(x_test)

print("x_test len =",len(x_test))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

price = model.predict(x_test)#,verbose=1)
price = scaler.inverse_transform(price)

rms = np.sqrt(np.mean(np.power((valid-price),2)))

print('Train Score: %.2f RMSE' % (rms))


print("Len of price",len(price))

for i in range(5):
    print("price = ",np.round(price[i],2)," Future Date of ", (Current_date + datetime.timedelta(i+1)))

# into hours, minutes and seconds 
import datetime 

def convert(n): 
	return str(datetime.timedelta(seconds = n)) 
	
n =  time.time() - start_time

def convert_sec(n): 
    return str(datetime.timedelta(seconds = n))

print("---Execution Time ---",convert_sec(n))