# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:28:46 2020

@author: Yu Zhe
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""
df = pd.read_csv("../monthly-car-sales.csv")

test = Predictor_HWES(df)
eval_result = test.evaluate_model()
print(eval_result)
result = test.outsample_forecast()
result.plot()
"""
#%%
"""
Holt Winter’s Exponential Smoothing (HWES)
"""
#from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

class Predictor_HWES:
    def __init__(self, df):
        df.columns = ['ds', 'y']
        df['ds']= pd.to_datetime(df['ds'])
        self.df = df.set_index('ds')
        self.y = self.df['y']


    """
    Fit ARIMA Model
    """
    def fit_model(self, df):
        param=['add', 'mul']
        #model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12, damped=True)
        #hw_model = model.fit(optimized=True, use_boxcox=False, remove_bias=False)
        model = ExponentialSmoothing(df, seasonal='add', seasonal_periods=12)
        model_fit = model.fit()
        return model_fit
        
    
    """
    Evaluate Forecast Model
    """
    def evaluate_model(self, output=True):
        # Predict last 12 months of data
        train = self.df[:-12]
        y_truth = self.y[-12:]
        model_fit = self.fit_model(train)
        y_forecasted = model_fit.predict(len(train), len(train)+11)
        
        mse = ((y_forecasted - y_truth) ** 2).mean()
        rounded_mse = round(mse, 2)
        rmse = round(np.sqrt(mse), 2)
        
        # calculate MAE between expected and predicted values
        mae = mean_absolute_error(y_truth, y_forecasted)

        if (output==True):
            print('The Mean Squared Error of our forecasts is {}'.format(rounded_mse))
            print('The Root Mean Squared Error of our forecasts is {}'.format(rmse))    
            print('MAE: %.3f' % mae)
            print(model_fit.summary()) 

        return rounded_mse, rmse, mae
    
    
    """
    Out-of-Sample Forecast
    Producing and visualizing forecasts
    """
    def outsample_forecast(self, output=True):
        model_fit = self.fit_model(self.df)
        yhat = model_fit.predict(len(self.df), len(self.df)+11)
        if (output==True):
            print(yhat)
            yhat.plot()
            
        return yhat