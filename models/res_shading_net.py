import tensorflow as tf
from tensorflow import keras

def ResShadingNet(input_shape=(512, 512, 6), weights_dir=None):

	def residual_block(y, channels, add_channels, dilation_rate=1):
		res_conv1 = keras.layers.Conv2D(channels, (3,3), dilation_rate=dilation_rate, strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(y)
		res_conv1 = keras.layers.BatchNormalization(axis=-1)(res_conv1)
		res_conv1 = keras.layers.LeakyReLU(0.01)(res_conv1)
		
		res_conv2 = keras.layers.Conv2D(channels, (3,3), strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(res_conv1)
		res_conv2 = keras.layers.BatchNormalization(axis=-1)(res_conv2)
		res_conv2 = keras.layers.LeakyReLU(0.01)(res_conv2)
		
		res_concat = keras.layers.Add()([y, res_conv2]) if add_channels else keras.layers.Concatenate()([y, res_conv2])
		
		return res_concat

	input_image = keras.Input(shape=input_shape)

	down_level_0_conv = keras.layers.Conv2D(16, (3, 3), strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(input_image)
	down_level_0_conv = keras.layers.LeakyReLU(0.01)(down_level_0_conv)
	down_level_0_to_1 = keras.layers.MaxPooling2D((2, 2))(down_level_0_conv)

	down_level_1_conv = keras.layers.Conv2D(32, (3, 3), strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(down_level_0_to_1) #256
	down_level_1_conv = keras.layers.BatchNormalization(axis=-1)(down_level_1_conv)
	down_level_1_conv = keras.layers.LeakyReLU(0.01)(down_level_1_conv)
	down_level_1_to_2 = keras.layers.MaxPooling2D((2, 2))(down_level_1_conv)

	down_level_2_conv = keras.layers.Conv2D(64, (3, 3), strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(down_level_1_to_2) # 128
	down_level_2_conv = keras.layers.BatchNormalization(axis=-1)(down_level_2_conv)
	down_level_2_conv = keras.layers.LeakyReLU(0.01)(down_level_2_conv)
	down_level_2_to_3 = keras.layers.MaxPooling2D((2, 2))(down_level_2_conv)

	block_1 = residual_block(down_level_2_to_3, 64, True)
	block_2 = residual_block(block_1, 64, True)
	block_3 = residual_block(block_2,  64, True)
	block_4 = residual_block(block_3,  64, False)
	block_5 = residual_block(block_4,  64, False)
	block_6 = residual_block(block_5,  64, False)
	block_7 = residual_block(block_6,  64, False)

	pyramid_level_1 = keras.layers.Conv2D(16, (1, 1), strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(block_7) # 128
	pyramid_level_3 = keras.layers.Conv2D(16, (3, 3), strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(block_7) # 128
	pyramid_level_5 = keras.layers.Conv2D(16, (5, 5), strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(block_7) # 128
	pyramid_level_11 = keras.layers.Conv2D(16, (11, 11), strides=1, padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(42))(block_7) # 128

	res_concat = keras.layers.Concatenate()([block_7, pyramid_level_1, pyramid_level_3, pyramid_level_5, pyramid_level_11])

	up_level_3_to_2 = keras.layers.Conv2DTranspose(64, (4, 4), strides=2, padding='SAME')(res_concat) # 128
	up_concat_level_2 = keras.layers.Concatenate()([down_level_2_conv, up_level_3_to_2])
	up_level_2_conv = keras.layers.Conv2D(32, (3, 3), strides=1, padding='SAME', activation='relu')(up_concat_level_2)
	up_level_2_conv = keras.layers.BatchNormalization(axis=-1)(up_level_2_conv)
	up_level_2_conv = keras.layers.LeakyReLU(0.01)(up_level_2_conv)

	up_level_2_to_1 = keras.layers.Conv2DTranspose(32, (4, 4), strides=2, padding='SAME')(up_level_2_conv) # 256
	up_concat_level_1 = keras.layers.Concatenate()([down_level_1_conv, up_level_2_to_1])
	up_level_1_conv = keras.layers.Conv2D(16, (3, 3), strides=1, padding='SAME', activation='relu')(up_concat_level_1)
	up_level_1_conv = keras.layers.BatchNormalization(axis=-1)(up_level_1_conv)
	up_level_1_conv = keras.layers.LeakyReLU(0.01)(up_level_1_conv)

	up_level_1_to_0 = keras.layers.Conv2DTranspose(16, (4, 4), strides=2, padding='SAME')(up_level_1_conv) # 512
	up_concat_level_0 = keras.layers.Concatenate()([down_level_0_conv, up_level_1_to_0])

	output = keras.layers.Conv2D(1, (3, 3), strides=1, padding='SAME')(up_concat_level_0)
	output = keras.layers.LeakyReLU(0.01)(output)

	# Create model.
	model = keras.Model(input_image, output)

	# Load weights.
	if weights_dir:
		model.load_weights(weights_dir)

	return model