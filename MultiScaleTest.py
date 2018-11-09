import argparse
import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras

from Model import ShadingNet
from Model import ScaleWeightPredictor

IM_SIZE = 512

parser = argparse.ArgumentParser(description='')
parser.add_argument('-train_set', help='Train tfrecord.', required=True)
parser.add_argument('-val_set', help='Validation tfrecord.', required=True)
args = parser.parse_args()

# Define the model.
shading_net = ShadingNet(weights_dir='models/model_2018-09-22 12:25:23.132954.h5')
shading_net.trainable=False

compositor_net = ScaleWeightPredictor()

input_image = keras.Input(shape=(512, 512, 6))
down_level_1 = keras.layers.MaxPooling2D((2, 2))(input_image)
down_level_2 = keras.layers.MaxPooling2D((2, 2))(down_level_1)

shading_level_0 = shading_net(input_image)
shading_level_1 = shading_net(down_level_1)
shading_level_2 = shading_net(down_level_2)

shading_level_2_up = keras.layers.UpSampling2D(size=(2, 2))(shading_level_2)
compositor_1 = compositor_net([shading_level_1, shading_level_2_up])
compositor_0 = keras.layers.UpSampling2D(size=(2, 2))(compositor_1)
final = compositor_net([shading_level_0, compositor_0])

model = keras.Model(input_image, final)
model.summary()

# Compile the model.
lr = 0.00003
loss_function = 'mean_squared_error'
model.compile(optimizer=tf.train.AdamOptimizer(lr), loss=loss_function)

# Initilize the data.
featdef = { 'input': tf.FixedLenFeature(shape=[], dtype=tf.string),
           'ground_truth': tf.FixedLenFeature(shape=[], dtype=tf.string)}
          
def _parse_record(example_proto, clip=False):
    """Parse a single record into image, weather labels, ground labels"""
    example = tf.parse_single_example(example_proto, featdef)
    im = tf.decode_raw(example['input'], tf.float32)
    im = tf.reshape(im, (IM_SIZE, IM_SIZE, 6))
    gt = tf.decode_raw(example['ground_truth'], tf.float32)
    gt = tf.reshape(gt, (IM_SIZE, IM_SIZE, 1))

    return im, gt

# Construct a TFRecordDataset.
ds_train = tf.data.TFRecordDataset(args.train_set).map(_parse_record)
ds_train = ds_train.shuffle(1000).batch(4).repeat()

ds_validation = tf.data.TFRecordDataset(args.val_set).map(_parse_record)
ds_validation = ds_validation.batch(4).repeat()

# Define the callbacks.
time_now = datetime.datetime.now()

output_file_name = 'multiscale__lr_%s__loss_%s__%s' % (lr, loss_function, time_now)

callbacks = [
  # Write TensorBoard logs to `./logs` directory.
  tf.keras.callbacks.TensorBoard(log_dir='./tensorboard'),

  # Write log on CSV.
  tf.keras.callbacks.CSVLogger('./logs/%s_training.log' % (output_file_name))
]

# Fit the model.
history = model.fit(ds_train, epochs=10, steps_per_epoch=500, 
                    validation_data=ds_validation, validation_steps=15, 
                    callbacks=callbacks)

# Save weights.
model_json = model.to_json()
with open("./models/%s_model.json" % (output_file_name) , "w") as json_file:
    json_file.write(model_json)

model.save('./models/%s_model.h5' % (output_file_name))
print('[INFO] Model saved.')