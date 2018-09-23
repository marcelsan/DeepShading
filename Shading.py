import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
import datetime

IM_SIZE = 512

parser = argparse.ArgumentParser(description='')
parser.add_argument('-train_set', help='Train tfrecord.', required=True)
parser.add_argument('-val_set', help='Validation tfrecord.', required=True)
args = parser.parse_args()

#ToDo: create a class for the model
#      store all the hyperparameters to a file

# Define the model.
input_image = keras.Input(shape=(512, 512, 6))

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

model = keras.Model(input_image, up_level_0_conv)
model.summary()

# serialize model to JSON
model_json = model.to_json()
with open("./models/model.json", "w") as json_file:
    json_file.write(model_json)

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
model.save('./models/model_%s.h5' % (time_now))
print('[INFO] Model saved.')