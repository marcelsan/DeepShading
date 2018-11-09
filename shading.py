import argparse
import datetime

import numpy as np
import tensorflow as tf

from initialize import FLAGS
from losses import perceptual_loss, l1_loss
from models.shading_net import ShadingNet
from tensorflow import keras

train_set = '/media/marcelsantos/DATA/DeepShadingDataBase/train_512.tfrecord'
val_set = '/media/marcelsantos/DATA/DeepShadingDataBase/validation_512.tfrecord'

model = ShadingNet()
model.summary()

#=== Loss function formulation =================================================
lr = FLAGS.learningRate
loss_function = 'perceptual_loss'

model.compile(optimizer=tf.train.AdamOptimizer(lr), loss=l1_loss)

#=== Setup Data ================================================
featdef = { 'input': tf.FixedLenFeature(shape=[], dtype=tf.string),
           'ground_truth': tf.FixedLenFeature(shape=[], dtype=tf.string)}
          
def _parse_record(example_proto, clip=False):
    """Parse a single record into image, weather labels, ground labels"""
    example = tf.parse_single_example(example_proto, featdef)
    im = tf.decode_raw(example['input'], tf.float32)
    im = tf.reshape(im, (FLAGS.imSize, FLAGS.imSize, 6))
    gt = tf.decode_raw(example['ground_truth'], tf.float32)
    gt = tf.reshape(gt, (FLAGS.imSize, FLAGS.imSize, 1))

    return im, gt

# Construct a TFRecordDataset.
ds_train = tf.data.TFRecordDataset(train_set).map(_parse_record)
ds_train = ds_train.shuffle(1000).batch(16).repeat()

ds_validation = tf.data.TFRecordDataset(val_set).map(_parse_record)
ds_validation = ds_validation.batch(8).repeat()

#=== Run training loop ============================================================

# Define the callbacks.
time_now = datetime.datetime.now()
output_file_name = 'shading__lr_%s__loss_%s__%s' % (lr, loss_function, time_now)

callbacks = [
  # Write log on CSV.
  tf.keras.callbacks.CSVLogger('./results/logs/%s_training.log' % (output_file_name))
]

# Fit the model.
history = model.fit(ds_train, epochs=40, steps_per_epoch=500, 
                    validation_data=ds_validation, validation_steps=15, 
                    callbacks=callbacks)

#=== Shut down ===================================================================

# Save the model.
model_json = model.to_json()
with open("./results/models/%s_model.json" % (output_file_name) , "w") as json_file:
    json_file.write(model_json)

model.save('./results/models/%s_model.h5' % (output_file_name))
print('[INFO] Model saved.')