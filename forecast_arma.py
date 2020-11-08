# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 00:46:34 2020

@author: Yu Zhe
"""
import pandas as pd
import numpy as np 


#df = pd.read_csv("../monthly-car-sales.csv")
#test = Predictor_ARMA(df)
#eval_result = test.evaluate_model()
#print(eval_result)
#result = test.outsample_forecast()
#result.plot()
#%%
"""
Autoregressive Moving Average (ARMA)

The Autoregressive Moving Average (ARMA) method models the next step in the sequence 
as a linear function of the observations and resiudal errors at prior time steps.
It combines both Autoregression (AR) and Moving Average (MA) models.

The method is suitable for univariate time series without trend and seasonal components.
"""
#from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)

class Predictor_ARMA:
    def __init__(self, df):
        df.columns = ['ds', 'y']
        df['ds']= pd.to_datetime(df['ds'])
        self.df = df.set_index('ds')
        self.y = self.df['y']


    """
    Fit ARIMA Model
    """
    def fit_model(self, df):
        model = sm.tsa.ARMA(df, order=(2, 1))
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