import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer ('imSize', 512, 'Input image size.')

tf.app.flags.DEFINE_float 	('learningRate', 0.00009, 'Learning rate for ADAM.')