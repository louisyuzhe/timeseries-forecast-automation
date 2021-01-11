# Automated univariate time-series forecasting

Evaluation metric for learning model
1. Mean absolute error (MAE)
2. Mean squared error (MSE)
3. Root Mean Square Error (RMSE)
4. Mean absolute percentage error (MAPE)
5. Symmetric mean absolute percentage error (SMAPE)
6. Mean Forecast Error (MFE)
7. Normalized mean squared error (NMSE)
8. Theil's U statistic


```python
from forecast_recommender import forecast_recommender
import pandas as pd

df1 = pd.read_csv("../monthly-car-sales.csv")
recommender = forecast_recommender(df1)
result, model_name = recommender.auto_forecast(1)
```

    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ma.L1         -0.8372      0.066    -12.691      0.000      -0.967      -0.708
    ma.S.L12      -0.4687      0.114     -4.108      0.000      -0.692      -0.245
    sigma2      2.673e+06    4.8e+05      5.573      0.000    1.73e+06    3.61e+06
    ==============================================================================
    
    ==============================================
    Metrics for ARIMA Model:
    Mean Absolute Error: 1451.103
    Mean Squared Error: 18608.652
    Root Mean Squared Error: 1711.321
    Mean absolute percentage error: 7.871
    Scaled Mean absolute percentage error: 8.138
    Mean forecast error: 17939.99
    Normalised mean squared error: 0.192
    Theil's U statistic: 0.0
    ==============================================
    
    
    ==============================================
    Metrics for FBProphet Model:
    Mean Absolute Error: 1336.814
    Mean Squared Error: 18608.652
    Root Mean Squared Error: 1749.191
    Mean absolute percentage error: 7.187
    Scaled Mean absolute percentage error: 7.407
    Mean forecast error: 18094.991
    Normalised mean squared error: 0.2
    Theil_u_statistic: 0.0
    ==============================================
    
    
    ==============================================
    Metrics for HWES Model:
    Mean Absolute Error: 1589.052
    Mean Squared Error: 18608.652
    Root Mean Squared Error: 2197.325
    Mean absolute percentage error: 8.622
    Scaled Mean absolute percentage error: 9.334
    Mean forecast error: 17159.808
    Normalised mean squared error: 0.316
    Theil_u_statistic: 0.0
    ==============================================
    
    


```python
print("\nBest model for this dataset is ", model_name)
print(result)
```

    
    Best model for this dataset is  FbProphet
               ds          yhat
    0  1969-01-01  15396.026630
    1  1969-02-01  16165.838547
    2  1969-03-01  21369.121163
    3  1969-04-01  23494.793981
    4  1969-05-01  25004.900023
    5  1969-06-01  22263.278413
    6  1969-07-01  17926.533617
    7  1969-08-01  15754.482263
    8  1969-09-01  14262.071702
    9  1969-10-01  18697.917520
    10 1969-11-01  18592.832407
    11 1969-12-01  16406.359115
    
