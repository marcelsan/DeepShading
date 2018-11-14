import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer ('imSize', 512, 'Input image size.')

tf.app.flags.DEFINE_float 	('learningRate', 0.00009, 'Learning rate for ADAM.')

tf.app.flags.DEFINE_string  ('trainSetDir', '/media/marcelsantos/DATA/DeepShadingDataBase/train_512.tfrecord',
							'Directory for the training dataset.')

tf.app.flags.DEFINE_string  ('valSetDir', '/media/marcelsantos/DATA/DeepShadingDataBase/validation_512.tfrecord',
							'Directory for the validation dataset.')