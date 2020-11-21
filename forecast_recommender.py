# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 01:10:00 2020

@author: Yu Zhe

Automated univariate time-series forecasting
Example:
    auto_forecast(dataframe, evaluation_metric)

Evaluation metric for learning model
1. Mean absolute error (MAE)
2. Mean squared error (MSE)
3. Root Mean Square Error (RMSE)
4. Mean absolute percentage error (MAPE)
5. Symmetric mean absolute percentage error (SMAPE)
6. Mean Forecast Error (MFE)
7. Normalized mean squared error (NMSE)
8. Theil's U statistic
"""

from forecast_arima import Predictor_ARIMA
from forecast_prophet import Predictor_fbprophet
from forecast_hwes import Predictor_HWES
import pandas as pd

class forecast_recommender:
    """
    Format checking for input dataframe 
    """
    def __init__(self, df):
        if(len(df.columns) != 2):
            print("Dataframe is restricted to 1 datetime column (left) and 1 data column (Right)")
            return -1
        #Check if datetime column only contains datetime value
        if(not pd.to_datetime(df.iloc[:,0], errors='coerce').notnull().all()):
           print("Datetime column (Left) must contains only datetime value")
           return -1 
        #Check if data only contains numerical value
        if(not pd.to_numeric(df.iloc[:,1], errors='coerce').notnull().all()):
           print("Data column (Right) must contains only numerical value")
           return -1 
        self.df = df

    def auto_forecast(self, evaluate_metric):
        evaluator=1 # rmse is used to evaluate forecast model by default
        eval_metric_dict={}
        
        arima_result = Predictor_ARIMA(self.df)
        arima_result.fit_model()
        arima_result.evaluate_model(output=False)
        eval_metric_arima = arima_result.analyze_estimator(output=False, eval_metric=evaluate_metric)
        eval_metric_dict["arima"] = eval_metric_arima
        
        fbprophet_result = Predictor_fbprophet(self.df)
        eval_metric_fbprophet = fbprophet_result.evaluate_model(output=False, eval_metric=evaluate_metric)
        eval_metric_dict["fbprophet"] = eval_metric_fbprophet
        
        hwes_result = Predictor_HWES(self.df)
        hwes_result.param_selection(12) #Choose best config
        eval_metric_hwes = hwes_result.evaluate_model(output=False, eval_metric=evaluate_metric)
        eval_metric_dict["hwes"] = eval_metric_hwes
        
        # Determine best model using lowest rmse value
        if(evaluator==1): #rmse, use rmse_dict
            best_eval_metric = min(eval_metric_dict, key=lambda k: eval_metric_dict[k]) 
    
        if(best_eval_metric=='arima'):
            result_df = arima_result.outsample_forecast(output=False)
            return result_df, 'ARIMA'
        elif(best_eval_metric=='hwes'):
            result_df = hwes_result.outsample_forecast(output=False)
            return result_df, 'HWES'
        elif(best_eval_metric=='fbprophet'):
            result_df =fbprophet_result.outsample_forecast(output=False)
            return result_df[['ds','yhat']], 'FbProphet'
    