# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 23:51:34 2020

@author: Yu Zhe
"""
from forecast_arima import Predictor_arima
from forecast_prophet import Predictor_fbprophet

import pandas as pd

df = pd.read_csv("../monthly-car-sales.csv")
test = Predictor_arima(df)
test.fit_model()
test.evaluate_model(output=False)
test.analyze_estimator(output=False)
pred_result = test.outsample_forecast(output=False)
print(pred_result)
test.diagnostic_model()
#%%
test2 = Predictor_fbprophet(df)
print(test2.outsample_forecast(output=False))
print(test2.evaluate_model(output=False))
