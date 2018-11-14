import tensorflow as tf
from tensorflow import keras

def ResShadingNet(input_shape=(None, None, 6), weights_dir=None):

	def residual_block(y):
		res_conv1 = keras.layers.Conv2D(64, (3,3), strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(y)
		res_conv1 = keras.layers.LeakyReLU(0.01)(res_conv1)
		res_conv2 = keras.layers.Conv2D(64, (3,3), strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(res_conv1)
		res_conv2 = keras.layers.LeakyReLU(0.01)(res_conv2)
		res_concat = keras.layers.Add()([y, res_conv2])

		return res_concat

	input_image = keras.Input(shape=input_shape)

	conv1 = keras.layers.Conv2D(64, (3,3), strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(input_image)
	conv1 = keras.layers.LeakyReLU(0.01)(conv1)
	conv2 = keras.layers.Conv2D(64, (3,3), strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(conv1)
	conv2 = keras.layers.LeakyReLU(0.01)(conv2)

	block_1 = residual_block(conv2)
	block_2 = residual_block(block_1)
	block_3 = residual_block(block_2)
	block_4 = residual_block(block_3)
	block_5 = residual_block(block_4)
	block_6 = residual_block(block_5)
	block_7 = residual_block(block_6)
	block_8 = residual_block(block_7)
	block_9 = residual_block(block_8)
	block_10 = residual_block(block_9)
	block_11 = residual_block(block_10)
	block_12 = residual_block(block_11)
	block_13 = residual_block(block_12)
	block_14 = residual_block(block_13)
	block_15 = residual_block(block_14)
	block_16 = residual_block(block_15)
	block_17 = residual_block(block_16)

	output = keras.layers.Conv2D(1, (3, 3), strides=1, padding='SAME')(block_17)
	output = keras.layers.LeakyReLU(0.01)(output)

	# Create model.
	model = keras.Model(input_image, output)

	# Load weights.
	if weights_dir:
		model.load_weights(weights_dir)

	return model