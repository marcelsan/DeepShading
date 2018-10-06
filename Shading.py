import argparse
import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras

from Model import ShadingNet

# ToDO: store all the hyperparameters to a file
IM_SIZE = 512

parser = argparse.ArgumentParser(description='')
parser.add_argument('-train_set', help='Train tfrecord.', required=True)
parser.add_argument('-val_set', help='Validation tfrecord.', required=True)
args = parser.parse_args()

model = ShadingNet()
model.summary()

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.AdamOptimizer(0.0001),
              loss='mean_absolute_error',
              metrics=['accuracy'])

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

callbacks = [
  # Write TensorBoard logs to `./logs` directory.
  tf.keras.callbacks.TensorBoard(log_dir='./tensorboard'),

  # Write log on CSV.
  tf.keras.callbacks.CSVLogger('./logs/training_%s.log' % (time_now)),
]

# Fit the model.
history = model.fit(ds_train, epochs=35, steps_per_epoch=500, 
                    validation_data=ds_validation, validation_steps=15, 
                    callbacks=callbacks)

# Save the model.
# Serialize model to JSON
model_json = model.to_json()
with open("./models/model.json", "w") as json_file:
    json_file.write(model_json)

# Save weights.
model.save('./models/model_%s.h5' % (time_now))
print('[INFO] Model saved.')