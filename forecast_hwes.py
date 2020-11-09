# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:28:46 2020

@author: Yu Zhe
"""
import pandas as pd
import numpy as np
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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class Predictor_HWES:
    def __init__(self, df):
        df.columns = ['ds', 'y']
        df['ds']= pd.to_datetime(df['ds'])
        self.df = df.set_index('ds')
        self.y = self.df['y']
        self.best_config = None
        self.train = self.df[:-12].copy()
        self.test = self.df[-12:].copy()
        
    """
    Fit ARIMA Model
    """
    def fit_model(self, df):
        #model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12, damped=True)
        #hw_model = model.fit(optimized=True, use_boxcox=False, remove_bias=False)
        t,d,s,p,b,r = self.best_config
        # define model
        if (t == None):
            model = ExponentialSmoothing(df, trend=t, seasonal=s, seasonal_periods=p)
        else:
            model = ExponentialSmoothing(df, trend=t, damped_trend=d, seasonal=s, seasonal_periods=p)
        model = ExponentialSmoothing(df, seasonal='add', seasonal_periods=12)
        model_fit = model.fit()
        return model_fit
        
    
    """
    Evaluate Forecast Model
    """
    def evaluate_model(self, output=True):
        # Predict last 12 months of data
        y_truth = self.y[-12:].copy()
        model_fit = self.fit_model(self.train)
        y_forecasted = model_fit.predict(len(self.train), len(self.train)+11)
        
        # Return evaluation array
        model_evaluation = self.model_eval(y_truth, y_forecasted)

        if (output==True):
            print(model_fit.summary()) 
            
        return model_evaluation[1], model_evaluation[2], model_evaluation[0]
    
    
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
    
    
    """
    Parameter Selection for HWES model
    input = seasonal period
    """
    def param_selection(self, seasonality=12):
        cfg_list = self.exp_smoothing_configs(seasonal=[seasonality]) #[0,6,12]
        best_RMSE = np.inf
        best_config = []
        #t1 = d1 = s1 = p1 = b1 = r1 = ''
        
        for j in range(len(cfg_list)):
            #print(j)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                try:
                    cg = cfg_list[j]
                    #print(cg)
                    t,d,s,p,b,r = cg
            
                    # define model
                    if (t == None):
                        model = ExponentialSmoothing(self.train, trend=t, seasonal=s, seasonal_periods=p)
                    else:
                        model = ExponentialSmoothing(self.train, trend=t, damped_trend=d, seasonal=s, seasonal_periods=p)
                    # fit model
                    model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
                    # make one step forecast
                    y_forecast = model_fit.forecast(12)
                    rmse = np.sqrt(mean_squared_error(self.test,y_forecast))
                    if rmse < best_RMSE:
                        best_RMSE = rmse
                        best_config = cfg_list[j]
                except:
                    continue
        self.best_config = best_config


    """
    Parameter Selection for HWES model
    input = seasonal period
    """
    def exp_smoothing_configs(self, seasonal=[None]):
        models = list()
        # define config lists
        t_params = ['add', 'mul', None]
        d_params = [True, False]
        s_params = ['add', 'mul', None]
        p_params = seasonal
        b_params = [True, False]
        r_params = [True, False]
        # create config instances
        for t in t_params:
            for d in d_params:
                for s in s_params:
                    for p in p_params:
                        for b in b_params:
                            for r in r_params:
                                cfg = [t,d,s,p,b,r]
                                models.append(cfg)
        return models
    

    def model_eval(self, y, predictions):    
        # Mean absolute error (MAE)
        mae = mean_absolute_error(y, predictions)
    
        # Mean squared error (MSE)
        mse = mean_squared_error(y, predictions)
    
    
        # SMAPE is an alternative for MAPE when there are zeros in the testing data. It
        # scales the absolute percentage by the sum of forecast and observed values
        SMAPE = np.mean(np.abs((y - predictions) / ((y + predictions)/2))) * 100
    
    
        # Calculate the Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(y, predictions))
    
        # Calculate the Mean Absolute Percentage Error
        # y, predictions = check_array(y, predictions)
        MAPE = np.mean(np.abs((y - predictions) / y)) * 100
    
        # mean_forecast_error
        mfe = np.mean(y - predictions)
    
        # NMSE normalizes the obtained MSE after dividing it by the test variance. It
        # is a balanced error measure and is very effective in judging forecast
        # accuracy of a model.
    
        # normalised_mean_squared_error
        NMSE = mse / (np.sum((y - np.mean(y)) ** 2)/(len(y)-1))
    
    
        # theil_u_statistic
        # It is a normalized measure of total forecast error.
        error = y - predictions
        mfe = np.sqrt(np.mean(predictions**2))
        mse = np.sqrt(np.mean(y**2))
        rmse = np.sqrt(np.mean(error**2))
        theil_u_statistic =  rmse / (mfe*mse)
    
    
        # mean_absolute_scaled_error
        # This evaluation metric is used to over come some of the problems of MAPE and
        # is used to measure if the forecasting model is better than the naive model or
        # not.
    
        
        # Print metrics
        print('Mean Absolute Error:', round(mae, 3))
        print('Mean Squared Error:', round(mse, 3))
        print('Root Mean Squared Error:', round(rmse, 3))
        print('Mean absolute percentage error:', round(MAPE, 3))
        print('Scaled Mean absolute percentage error:', round(SMAPE, 3))
        print('Mean forecast error:', round(mfe, 3))
        print('Normalised mean squared error:', round(NMSE, 3))
        print('Theil_u_statistic:', round(theil_u_statistic, 3))
        
        return [mae,mse,rmse,MAPE,SMAPE,mfe,NMSE,theil_u_statistic]