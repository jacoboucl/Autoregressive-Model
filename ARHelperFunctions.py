import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

def CreateSeries(length=200, random_noise=0.1):
	'''
	This function creates a noisy sin wave to use as
	a time series to predict against
	'''
	
	series = np.sin(0.1*np.arange(length)) + np.random.randn(length)*random_noise
	
	return series
	
def BuildDataset(series, num_lags=10):
	'''
	This function creates a machine learning style data set for solving an 
	AR time series forecasting problem
	'''
	
	features = []
	target = []
	for t in range(len(series) - num_lags):
		x = series[t:t+num_lags]
		features.append(x)
		y = series[t+num_lags]
		target.append(y)

	X = np.array(features).reshape(-1, num_lags)
	Y = np.array(target)
	N = len(X)
	
	X_train, Y_train = X[:-N//2], Y[:-N//2]
	X_test, Y_test = X[-N//2:], Y[-N//2:]
	
	return X_train, Y_train, X_test, Y_test
	
def ModelBuild(T=10):
	'''
	This function builds an autogressive model
	'''
	
	i = Input(shape=(T,))
	x = Dense(1)(i)
	model = Model(i, x)
	
	return model
	
def IncorrectForecast(Y_test, X_test, model):
	'''
	
	'''
	
	validation_target = Y_test
	validation_predictions = []
	
	i = 0
	while len(validation_predictions) < len(validation_target):
		p = model.predict(X_test[i].reshape(1, -1))[0, 0]
		i += 1
  
		validation_predictions.append(p)
	
	return validation_predictions
	
def CorrectForecast(Y_test, X_train, model):
	'''
	
	'''
	
	validation_target = Y_test
	validation_predictions = []
	
	last_x = X_train[-1]

	while len(validation_predictions) < len(validation_target):
		p = model.predict(last_x.reshape(1, -1))[0, 0]
  
		validation_predictions.append(p)
  
		last_x = np.roll(last_x, -1)
		last_x[-1] = p
	
	return validation_predictions