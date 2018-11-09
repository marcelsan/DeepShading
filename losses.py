import keras.backend as K
import numpy as np

from initialize import FLAGS
from keras.applications.vgg16 import VGG16
from keras.models import Model

def perceptual_loss(y_true, y_pred): 
	vgg = VGG16(include_top=False, weights='imagenet', input_shape=(FLAGS.imSize, FLAGS.imSize, 3)) 	
	loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block1_conv2').output) 
	loss_model.trainable = False
    
	return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

def l1_loss(y_true, y_pred):
	return K.mean(K.abs(y_pred - y_true))

def l2_loss(y_true, y_pred):
	return K.mean(K.square(y_pred - y_true))

def gradient_loss(self, y_true, y_pred, averaged_samples):
	gradients = K.gradients(y_pred, averaged_samples)[0]
	gradients_sqr = K.square(gradients)
	gradients_sqr_sum = K.sum(gradients_sqr,
	                          axis=np.arange(1, len(gradients_sqr.shape)))	
	gradient_l2_norm = K.sqrt(gradients_sqr_sum)
	gradient_penalty = K.square(1 - gradient_l2_norm)	
	return K.mean(gradient_penalty)