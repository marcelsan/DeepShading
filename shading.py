import argparse
import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras

from models.shading_net import ShadingNet

# ToDO: store all the hyperparameters to a file
IM_SIZE = 512

parser = argparse.ArgumentParser(description='')
parser.add_argument('-train_set', help='Train tfrecord.', required=True)
parser.add_argument('-val_set', help='Validation tfrecord.', required=True)
args = parser.parse_args()

model = ShadingNet()
model.summary()

# The compile step specifies the training configuration.
lr = 0.00009
loss_function = 'mean_absolute_error'
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
ds_train = ds_train.shuffle(1000).batch(16).repeat()

ds_validation = tf.data.TFRecordDataset(args.val_set).map(_parse_record)
ds_validation = ds_validation.batch(8).repeat()

# Define the callbacks.
time_now = datetime.datetime.now()
output_file_name = 'shading__lr_%s__loss_%s__%s' % (lr, loss_function, time_now)

callbacks = [
  # Write log on CSV.
  tf.keras.callbacks.CSVLogger('./results/logs/%s_training.log' % (output_file_name))
]

# Fit the model.
history = model.fit(ds_train, epochs=45, steps_per_epoch=500, 
                    validation_data=ds_validation, validation_steps=15, 
                    callbacks=callbacks)

# Save the model.
model_json = model.to_json()
with open("./results/models/%s_model.json" % (output_file_name) , "w") as json_file:
    json_file.write(model_json)

model.save('./results/models/%s_model.h5' % (output_file_name))
print('[INFO] Model saved.')