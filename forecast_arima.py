# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 00:23:22 2020

@author: Yu Zhe
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#df = pd.read_csv("monthly-car-sales.csv")

#%%
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('once', category=UserWarning)
import itertools
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error

class Predictor_arima:
    def __init__(self, df):
        df.columns = ['ds', 'y']
        df['ds']= pd.to_datetime(df['ds'])
        self.df = df.set_index('ds')
        self.y = self.df['y']
        self.optimal_param = self.param_selection()
        self.results = None
        self.pred = None
        
    """
    Time-series decomposition 
    Decompose time series into three distinct components: trend, seasonality, and noise
    """
    def decompose_data(self):
        decomposition = sm.tsa.seasonal_decompose(self.df, model='additive')
        fig = decomposition.plot()
        fig.show()
    
    
    """
    Parameter Selection for ARIMA model
    p, d, q = seasonality, trend, noise

    """
    def param_selection(self):
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        
        #parameter combinations for ARIMA
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        model_result=pd.DataFrame(columns=['param', 'param_seasonal', 'aic'])
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    try:
                        mod = sm.tsa.statespace.SARIMAX(self.y,
                                                        order=param,
                                                        seasonal_order=param_seasonal,
                                                        enforce_stationarity=False,
                                                        enforce_invertibility=False)
                        with warnings.catch_warnings():
                            warnings.filterwarnings('error')
                            results = mod.fit()
                        newrow = [{'param':param,'param_seasonal':param_seasonal,'aic':results.aic}]
                        model_result=model_result.append(newrow,ignore_index=True,sort=False)
                        #model_result.append([param, param_seasonal, results.aic])
                    except:
                        continue
        
        # Lowest AIC = optimal set of parameters that yields the best performance for ARIMA model 
        #print(model_result[model_result.aic == model_result.aic.min()])
        optimal_param = model_result[model_result.aic == model_result.aic.min()]
        #print(model_result)
        return optimal_param
        
    
    """
    Fit ARIMA Model
    """
    def fit_model(self):
        mod = sm.tsa.statespace.SARIMAX(self.y,
                                        order=self.optimal_param.param.values[0],
                                        seasonal_order=self.optimal_param.param_seasonal.values[0],
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
        self.results = mod.fit()
        print(self.results.summary().tables[1])


    """
    Model diagnostics
    """
    def diagnostic_model(self):
        self.results.plot_diagnostics(figsize=(16, 8))
        plt.show()        
        
    
    """
    Evaluate Forecast Model
    """
    def evaluate_model(self, output=True):
        # Predict last 12 months of data
        self.pred = self.results.get_prediction(start=-12, dynamic=False)
        pred_ci = self.pred.conf_int()
        if (output==True):
            ax = self.y.plot(label='observed')
            self.pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
            ax.fill_between(pred_ci.index,
                            pred_ci.iloc[:, 0],
                            pred_ci.iloc[:, 1], color='k', alpha=.2)
            ax.set_xlabel('Date')
            ax.set_ylabel('Furniture Sales')
            plt.legend()
            plt.show()
        
    """
    Analyze quality of the estimator 
    """
    def analyze_estimator(self, output=True):
        y_forecasted = self.pred.predicted_mean
        y_truth = self.y[-12:]
        mse = ((y_forecasted - y_truth) ** 2).mean()
        rounded_mse = round(mse, 2)
        rmse = round(np.sqrt(mse), 2)
        
        # calculate MAE between expected and predicted values
        mae = mean_absolute_error(y_truth, y_forecasted)

        if (output==True):
            print('The Mean Squared Error of our forecasts is {}'.format(rounded_mse))
            print('The Root Mean Squared Error of our forecasts is {}'.format(rmse))    
            print('MAE: %.3f' % mae)
        
        return rounded_mse, rmse, mae
    
    """
    Out-of-Sample Forecast
    Producing and visualizing forecasts
    """
    def outsample_forecast(self, output=True):
        pred_uc = self.results.get_forecast(steps=12)
        pred_ci = pred_uc.conf_int()
        if (output==True):
            ax = self.y.plot(label='observed', figsize=(14, 7))
            pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
            ax.fill_between(pred_ci.index,
                            pred_ci.iloc[:, 0],
                            pred_ci.iloc[:, 1], color='k', alpha=.25)
            ax.set_xlabel('Date')
            ax.set_ylabel('count')
            plt.legend()
            plt.show()
        return pred_uc.predicted_mean
        
#%%
#Test functions
"""
test = Predictor_model2(df)
#test.decompose_data()
test.fit_model()
test.evaluate_model(output=False)
test.analyze_estimator(output=False)
pred_result = test.outsample_forecast(output=False)
print(pred_result)
"""