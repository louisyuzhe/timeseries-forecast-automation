# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 23:51:34 2020

@author: Yu Zhe
"""
from forecast_recommender import forecast_recommender
import pandas as pd
df = pd.read_csv("../monthly-car-sales.csv")

"""
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

df1 = pd.read_csv("../monthly-car-sales.csv")
recommender = forecast_recommender(df1)
result, model_name = recommender.auto_forecast(0)
print(model_name)
print(result)
