import numpy as np
import tensorflow as tf
from tensorflow import keras

def ShadingNet(input_shape=(None, None, 6), weights_dir=None):

	input_image = keras.Input(shape=input_shape)

	# Branch going down.
	down_level_0_conv = keras.layers.Conv2D(8, (3, 3), strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(input_image)
	down_level_0_conv = keras.layers.LeakyReLU(0.01)(down_level_0_conv)
	down_level_0_to_1 = keras.layers.MaxPooling2D((2, 2))(down_level_0_conv)

	down_level_1_conv = keras.layers.Conv2D(16, (3, 3), strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(down_level_0_to_1) #256
	down_level_1_conv = keras.layers.LeakyReLU(0.01)(down_level_1_conv)
	down_level_1_to_2 = keras.layers.MaxPooling2D((2, 2))(down_level_1_conv)

	down_level_2_conv = keras.layers.Conv2D(32, (3, 3), strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(down_level_1_to_2) # 128
	down_level_2_conv = keras.layers.LeakyReLU(0.01)(down_level_2_conv)
	down_level_2_to_3 = keras.layers.MaxPooling2D((2, 2))(down_level_2_conv)

	down_level_3_conv = keras.layers.Conv2D(64, (3, 3), strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(down_level_2_to_3) # 64
	down_level_3_conv = keras.layers.LeakyReLU(0.01)(down_level_3_conv)
	down_level_3_to_4 = keras.layers.MaxPooling2D((2, 2))(down_level_3_conv)

	down_level_4_conv = keras.layers.Conv2D(128, (3, 3), strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(down_level_3_to_4) # 32
	down_level_4_conv = keras.layers.LeakyReLU(0.01)(down_level_4_conv)
	down_level_4_to_5 = keras.layers.MaxPooling2D((2, 2))(down_level_4_conv)
	down_level_4_to_5 = keras.layers.Dropout(0.5)(down_level_4_to_5)

	down_level_5_conv = keras.layers.Conv2D(256, (3, 3), strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(down_level_4_to_5) # 16
	down_level_5_conv = keras.layers.LeakyReLU(0.01)(down_level_5_conv)
	down_level_5_conv = keras.layers.Dropout(0.5)(down_level_5_conv)

	# Branch going up.
	up_level_5_to_4 = keras.layers.Conv2DTranspose(256, (4, 4), strides=2, padding='SAME')(down_level_5_conv) # 32
	up_concat_level_4 = keras.layers.Concatenate()([down_level_4_conv, up_level_5_to_4])
	up_level_4_conv = keras.layers.Conv2D(128, (3, 3), strides=1, padding='SAME', activation='relu')(up_concat_level_4) 
	up_level_4_conv = keras.layers.LeakyReLU(0.01)(up_level_4_conv)
	up_level_4_conv = keras.layers.Dropout(0.5)(up_level_4_conv)

	up_level_4_to_3 = keras.layers.Conv2DTranspose(128, (4, 4), strides=2, padding='SAME')(up_level_4_conv) # 64
	up_concat_level_3 = keras.layers.Concatenate()([down_level_3_conv, up_level_4_to_3])
	up_level_3_conv = keras.layers.Conv2D(64, (3, 3), strides=1, padding='SAME', activation='relu')(up_concat_level_3) 
	up_level_3_conv = keras.layers.LeakyReLU(0.01)(up_level_3_conv)
	up_level_3_conv = keras.layers.Dropout(0.5)(up_level_3_conv)

	up_level_3_to_2 = keras.layers.Conv2DTranspose(64, (4, 4), strides=2, padding='SAME')(up_level_3_conv) # 128
	up_concat_level_2 = keras.layers.Concatenate()([down_level_2_conv, up_level_3_to_2])
	up_level_2_conv = keras.layers.Conv2D(32, (3, 3), strides=1, padding='SAME', activation='relu')(up_concat_level_2)
	up_level_2_conv = keras.layers.LeakyReLU(0.01)(up_level_2_conv)

	up_level_2_to_1 = keras.layers.Conv2DTranspose(32, (4, 4), strides=2, padding='SAME')(up_level_2_conv) # 256
	up_concat_level_1 = keras.layers.Concatenate()([down_level_1_conv, up_level_2_to_1])
	up_level_1_conv = keras.layers.Conv2D(16, (3, 3), strides=1, padding='SAME', activation='relu')(up_concat_level_1)
	up_level_1_conv = keras.layers.LeakyReLU(0.01)(up_level_1_conv)

	up_level_1_to_0 = keras.layers.Conv2DTranspose(16, (4, 4), strides=2, padding='SAME')(up_level_1_conv) # 512
	up_concat_level_0 = keras.layers.Concatenate()([down_level_0_conv, up_level_1_to_0])
	up_level_0_conv = keras.layers.Conv2D(1, (3, 3), strides=1, padding='SAME', activation='relu')(up_concat_level_0)
	up_level_0_conv = keras.layers.LeakyReLU(0.01)(up_level_0_conv)

	# Create model.
	model = keras.Model(input_image, up_level_0_conv)

	# Load weights.
	if weights_dir:
		model.load_weights(weights_dir)

	return model