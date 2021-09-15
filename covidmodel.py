import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from datetime import datetime 
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',FutureWarning)
from math import sqrt

import pickle

#########Load the Data Set csv file ################# 
data =pd.read_csv("Beds_Occupied.csv")

# to get the total beds available
data['Total Beds Available']=900-data['Total Inpatient Beds']
data.head()

data = data.drop("Total Inpatient Beds",axis=1)

data.collection_date = pd.to_datetime(data.collection_date, format= "%d-%m-%Y")


data.set_index('collection_date',inplace=True)


dt = pd.date_range("06-15-2020","06-15-2021")
idx = pd.DatetimeIndex(dt)
data= data.reindex(idx)


pd.set_option('display.max_rows', data.shape[0]+1)


data =data.interpolate(method='time')


train = data.head(336)
test = data.tail(30)



y_hat_hwm = test.copy()
holt_win_model = ExponentialSmoothing(np.asarray(train['Total Beds Available']) ,seasonal_periods=7 ,trend='add', seasonal='add')
holt_win_model = holt_win_model.fit(optimized=True)
y_hat_hwm['hw_forecast'] = holt_win_model.forecast(30)


pickle.dump(holt_win_model, open("covidmodel.pkl", "wb"))


