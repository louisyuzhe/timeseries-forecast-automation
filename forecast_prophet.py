# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:23:14 2020

@author: ylim
"""
from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np 
import logging, sys
logging.disable(sys.maxsize)
import warnings
warnings.filterwarnings("ignore")
#dataset = pd.read_csv('monthly-car-sales.csv')
#%%
#Test functions
"""
a = Predictor_fbprophet(dataset)
a.outsample_forecast()
print(a.outsample_forecast(output=False))
a.evaluate_model()
print(a.evaluate_model(output=False))
"""
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
        model = Prophet(interval_width=0.95)
        
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
        #future = model.make_future_dataframe(periods=step, freq='MS')
        # use the model to make a forecast
        forecast = model.predict(future)
        
        return forecast, model
    
    """
    Out-of-Sample Forecast
    """
    def outsample_forecast(self, output=True):
        forecast, model = self.forecast(self.df, 12)
        
        if (output==True):
            # summarize the forecast
            print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
            
            # plot forecast
            model.plot(forecast)
            model.plot_components(forecast)
            plt.show()
        
        return forecast
            
    """
    Evaluate Forecast Model
    Default evaluation metric is RMSE
    """
    def evaluate_model(self, output=True, eval_metric=2):
        # Last 12 months as test data
        train = self.df.drop(self.df.index[-12:])
        
        #Fit training dataset
        forecast, model = self.forecast(train, 12)
        # calculate MAE between expected and predicted values
        y_true = self.df[self.df.columns[1]][-12:].values
        y_pred = forecast['yhat'].values
        """
        mae = mean_absolute_error(y_true, y_pred)
        
        
        mse = ((y_pred - y_true) ** 2).mean()
        rounded_mse = round(mse, 2)
        rmse = round(np.sqrt(mse), 2)
        """
        if (output==True):
            """
            print('MAE: %.3f' % mae)
            print('The Mean Squared Error of our forecasts is {}'.format(rounded_mse))
            print('The Root Mean Squared Error of our forecasts is {}'.format(rmse))
            """
            # plot expected vs actual
            plt.plot(forecast['ds'], y_true, label='Actual')
            plt.plot(forecast['ds'], y_pred, label='Predicted')
            plt.xticks(forecast['ds'], rotation=65)
            plt.legend()
            plt.show()
        """
        plt.figure(figsize=(10, 7))
        plt.plot(forecast['Date'], forecast['y_trend'], 'b-')
        plt.legend(); plt.xlabel('Date'); plt.ylabel('y')
        plt.title('y Trend');
        """
        # Return evaluation array
        model_evaluation = self.model_eval(y_true, y_pred)
        
        #return rounded_mse, rmse, mae
        #print(model_evaluation[1], model_evaluation[2], model_evaluation[0])
        return model_evaluation[eval_metric]


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
        print("\n==============================================")
        print("Metrics for FBProphet Model:")
        print('Mean Absolute Error:', round(mae, 3))
        print('Mean Squared Error:', round(mse, 3))
        print('Root Mean Squared Error:', round(rmse, 3))
        print('Mean absolute percentage error:', round(MAPE, 3))
        print('Scaled Mean absolute percentage error:', round(SMAPE, 3))
        print('Mean forecast error:', round(mfe, 3))
        print('Normalised mean squared error:', round(NMSE, 3))
        print('Theil_u_statistic:', round(theil_u_statistic, 3))
        print("==============================================\n")

        return [mae,mse,rmse,MAPE,SMAPE,mfe,NMSE,theil_u_statistic]