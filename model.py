import numpy as np 
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from sklearn.metrics import mean_squared_error 

import os
from joblib import dump 

def convert_timeseries_stationary(data, alpha = 0.05, max_diff = 10):
    if adfuller(data)[1] < alpha:
        return {'diff_order':0,
                'time_series': data}
    
    p_values = [(i,adfuller(data.diff(i).dropna())[1]) for i in range(1,max_diff)]

    significant = [p for p in p_values if p[1] < alpha]
    significant = sorted(significant, key=lambda x: x[1])

    diff_order = significant[0][0]
    ts = data.diff(diff_order).dropna()
    return {'diff_order': diff_order,
            'time_series': ts.resample('M').mean()}


def GridSearchARIMA(ts,d,train_per=0.7):
    """ GridSearchARIMA recieves stationary time-series ts with d the number of time
        the time series was differenced. We split the time series into train and test data and then proceed to compute ARIMA model on the train data , for different values of p,q with d and seasonal parameters fixed ,and compare the one with the best RMSE value on the test data. 
        We return the ARIMA model with the best parameters trained on the whole dataset.
    """
    train_len = int(train_per*len(ts))
    start,end = ts.index.values[train_len], ts.index.values[-1]
    params = [(1,2),(2,3),(4,1)]
    minimal_rmse = np.inf
    best_model = None 
    best_params = None 
    for param in params:
        model = SARIMAX(ts[:train_len],
                        order=(param[0],d,param[1]),
                        seasonal_order=(1,0,0,12))
        result = model.fit()
        start,end = ts.index.values[train_len], ts.index.values[-1]
        predictions = result.predict(start=start,end=end)
        rmse = mean_squared_error(ts[start:end],predictions)
        if rmse < minimal_rmse:
            minimal_rmse = rmse 
            best_params = param 

    best_model = SARIMAX(ts,
                         order=(best_params[0],d,best_params[1]),
                         seasonal_order=(1,0,0,12))
    result = best_model.fit()
    return result