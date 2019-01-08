import keras.backend as K
import numpy as np

def l1_loss(y_true, y_pred):
	return K.mean(K.abs(y_pred - y_true))

def l2_loss(y_true, y_pred):
	return K.mean(K.square(y_pred - y_true))