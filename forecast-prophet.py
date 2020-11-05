# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:23:14 2020

@author: ylim
"""
from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import numpy as np 

dataset = pd.read_csv('monthly-car-sales.csv')
#%%
a = Predictor_fbprophet(dataset)
a.outsample_forecast()
a.evaluate_model()
#%%
class Predictor_fbprophet:
    def __init__(self, df):
        self.df = df
        self.model = None


    """
    Fit Prophet Model
    """
    def fit_model(self, df):
        # prepare expected column names
        df.columns = ['ds', 'y']
        df['ds']= pd.to_datetime(df['ds'])
        
        # define the model
        model = Prophet()
        
        # fit the model
        model.fit(df)
        
        return model


    """
    Forecast next step
    """
    def forecast(self, df, step):
        model = self.fit_model(df)
        # define the period for next step(s)
        lastDate= pd.Timestamp(df['ds'][len(df)-1])
        
        future = list()
        for i in range(1, step+1):

            lastDate+= pd.DateOffset(months=1)
            future.append([lastDate])        
        future = pd.DataFrame(future)
        future.columns = ['ds']
        future['ds']= pd.to_datetime(future['ds'])
        
        # use the model to make a forecast
        forecast = model.predict(future)
        
        return forecast, model
    
    """
    Out-of-Sample Forecast
    """
    def outsample_forecast(self):
        forecast, model = self.forecast(self.df, 12)
        
        # summarize the forecast
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        
        # plot forecast
        model.plot(forecast)
        plt.show()
            
    """
    Evaluate Forecast Model
    """
    def evaluate_model(self):
        # Last 12 months as test data
        train = self.df.drop(self.df.index[-12:])
        
        #Fit training dataset
        forecast, model = self.forecast(train, 12)

        # calculate MAE between expected and predicted values for december
        y_true = self.df[self.df.columns[1]][-12:].values
        y_pred = forecast['yhat'].values
        mae = mean_absolute_error(y_true, y_pred)
        print('MAE: %.3f' % mae)
        
        mse = ((y_pred - y_true) ** 2).mean()
        rounded_mse = round(mse, 2)
        rmse = round(np.sqrt(mse), 2)
        print('The Mean Squared Error of our forecasts is {}'.format(rounded_mse))

        print('The Root Mean Squared Error of our forecasts is {}'.format(rmse))
        
        # plot expected vs actual
        plt.plot(forecast['ds'], y_true, label='Actual')
        plt.plot(forecast['ds'], y_pred, label='Predicted')
        plt.xticks(forecast['ds'], rotation=65)
        plt.legend()
        plt.show()
        
        return rounded_mse, rmse, mae
