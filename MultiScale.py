import argparse
import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras

from Model import ShadingNet

shading_net = ShadingNet(weights_dir='models/model_2018-09-22 12:25:23.132954.h5')



# Compile the model.
model.compile(optimizer=tf.train.AdamOptimizer(0.0001),
              loss='mean_absolute_error',
              metrics=['accuracy'])