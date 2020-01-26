# Autoregressive-Model
Simple implementation of an auto regressive model using Tensorflow 2.0

## Base Series
A simple noisy sine wave was created as the base series on which we will attempt to build and fit an auto regressive model.

![Noisy AR Series](/images/noisy_ar_series.png)

## Model Architecture

## Training Logs
![Training Logs](/images/training_logs.png)

## Incorrect Forecast
The below figure represents the incorrect method for forecasting time series data, where the model only predicts one timestep into the future at a time. 
Before reverting back to using the known feature values for the next time steps.

![Incorrect Forecast](/images/incorrect_forecast.png)

## Correct Forecast
The below figure represents the correct method for forecasting time series data, where the model is forced to use its previous predictions to forecast across 
more than one time step. 

![Correct Forecast](/images/correct_forecast.png)