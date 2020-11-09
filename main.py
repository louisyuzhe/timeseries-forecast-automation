# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 23:51:34 2020

@author: Yu Zhe
"""
from forecast_arima import Predictor_ARIMA
from forecast_prophet import Predictor_fbprophet
from forecast_arma import Predictor_ARMA
from forecast_hwes import Predictor_HWES

import pandas as pd
df = pd.read_csv("../monthly-car-sales.csv")

#%%
"""
ARIMA
"""
arima_result = Predictor_ARIMA(df)
arima_result.fit_model()
arima_result.evaluate_model(output=False)
arima_result.analyze_estimator(output=False)
arima_pred = arima_result.outsample_forecast(output=False)
print(arima_pred)
#arima_pred.plot()
arima_result.diagnostic_model()

#%%
"""
Fbprophet
"""
fbprophet_result = Predictor_fbprophet(df)
fbprophet_pred = fbprophet_result.outsample_forecast(output=False)
print(fbprophet_pred)
print(fbprophet_result.evaluate_model(output=False))
fbprophet_pred_yhat = fbprophet_pred['yhat']
fbprophet_pred_yhat.index = fbprophet_pred['ds']

#%%
"""
ARMA
"""
arma_result = Predictor_ARMA(df)
arma_result.evaluate_model(output=False)
print(arma_result)
arma_pred = arma_result.outsample_forecast(output=False)
#arma_pred.plot()
arma_pred+=fbprophet_seasonality

#%%
"""
HWES
"""
hwes_result = Predictor_HWES(df)
hwes_result.param_selection(12) #Choose best config
mse,rmse,mae=hwes_result.evaluate_model(output=False)
print(mse,rmse,mae)
hwes_pred = hwes_result.outsample_forecast(output=False)
print(hwes_pred)
#arma_pred.plot()

#%%
"""
Comparison
"""

original_df = df.set_index('ds')
ax = original_df['y'].plot(label= 'Observed', color='cyan', legend=True)

arima_pred.plot(label= 'ARIMA_forecast', color='blue', legend=True, ax=ax)
fbprophet_pred_yhat.plot(label= 'fbprophet_forecast', color='red', legend=True, ax=ax)
arma_pred.plot(label= 'ARMA_forecast', color='green', legend=True, ax=ax)
hwes_pred.plot(label= 'HWES_forecast', color='orange', legend=True, ax=ax)

#%%
"""
Get Seasonality
"""
fbprophet_seasonality = fbprophet_pred['additive_terms']
fbprophet_seasonality.index = fbprophet_pred['ds']
fbprophet_seasonality.plot()
