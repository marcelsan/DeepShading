import numpy as np
import tensorflow as tf
from tensorflow import keras

def ScaleWeightPredictor():

	def blend_func(x):
		i_f = x[0]
		i_c = x[1]
		alpha = x[2]

		return i_f - alpha * i_f + alpha * i_c

	fine_image = keras.Input(shape=(None, None, 1))
	coarse_image = keras.Input(shape=(None, None, 1))

	# Scale-weight predictor block.
	input_concat = keras.layers.Concatenate()([fine_image, coarse_image])

	# 1x1 convolution.
	conv1 = keras.layers.Conv2D(64, 1, activation='relu', strides=1, kernel_initializer=tf.keras.initializers.glorot_uniform(42))(input_concat)

	# Residual blocks.
	res_1_conv1 = keras.layers.Conv2D(64, (5, 5), activation='relu', strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(conv1)
	res_1_conv2 = keras.layers.Conv2D(64, (5, 5), activation='relu', strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(res_1_conv1)
	res_1_concat = keras.layers.Concatenate(name='Concatenate_Res_1')([conv1, res_1_conv2])
	res_2_conv1 = keras.layers.Conv2D(64, (5, 5), activation='relu', strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(res_1_concat)
	res_2_conv2 = keras.layers.Conv2D(64, (5, 5), activation='relu', strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(res_2_conv1)
	res_2_concat = keras.layers.Concatenate(name='Concatenate_Res_2')([res_1_concat, res_2_conv2])

	# 1x1 convolution.
	#conv2 = keras.layers.Conv2D(64, 1, activation='relu', strides=1, kernel_initializer=tf.keras.initializers.glorot_uniform(42))(res_2_concat)

	alpha = keras.layers.Conv2D(1, 1, activation='sigmoid', strides=1, kernel_initializer=tf.keras.initializers.glorot_uniform(42))(res_2_concat)

	# Scale blender. 
	final = keras.layers.Lambda(blend_func, output_shape=(None, None, 1))([fine_image, coarse_image, alpha])

	model = keras.Model([fine_image, coarse_image], final)

	return model