# -*- coding: utf-8 -*-
"""AR-Main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xh2Vjj4u5zpAv97ssE6UpFKDDCUhz72Y
"""

# install tensorflow
!pip install -q tensorflow==2.0.0-beta1
import tensorflow as tf

# check tensorflow version
print(tf.__version__)

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# upload a python file with some useful functions
from google.colab import files
uploaded = files.upload()

from ARHelperFunctions import *

series = create_series(length=180, random_noise=0.05)

# plot noisy series
plt.plot(series)
plt.title("Noisy Auto-Regressive Series")
plt.show()

X_train, Y_train, X_test, Y_test = build_dataset(series, num_lags=10)

model = model_build(T=10)

model.compile(loss="mse", optimizer=Adam(lr=0.1))

training_logs = model.fit(X_train, Y_train, 
                          epochs=200, validation_data=(X_test, Y_test),
                          verbose=0)

# plot training logs
plt.plot(training_logs.history["loss"], label="Loss")
plt.plot(training_logs.history["val_loss"], label="Validation Loss")
plt.title("Training Logs")
plt.legend()

validation_target = Y_test

incorrect_validation_predictions = incorrect_forecast(Y_test, X_test, model)

# plot results of incorrect forecasting method
plt.plot(validation_target, label="Target")
plt.plot(incorrect_validation_predictions, label="Prediction")
plt.title("Incorrect Forecast")
plt.legend()

correct_validation_predictions = correct_forecast(Y_test, X_train, model)

# plot results of correct forecasting method
plt.plot(validation_target, label="Target")
plt.plot(correct_validation_predictions, label="Prediction")
plt.title("Correct Forecast")
plt.legend()