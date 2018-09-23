import numpy as np
import tensorflow as tf
from tensorflow import keras
import pyexr

json_file = open('./models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('./models/model_2018-09-22 12:25:23.132954.h5')
print("[INFO] Loaded model from disk")
 
# Evaluate loaded model on test data
loaded_model.compile(optimizer=tf.train.AdamOptimizer(0.0001),
              loss='mean_absolute_error',
              metrics=['accuracy'])

normal_image = pyexr.read_all('/media/marcelsantos/StorageDevice/DeepShadingValidationImages/Sibenik/Normals/0000000001.exr')['default'][:,:,0:3]
position_image = pyexr.read_all('/media/marcelsantos/StorageDevice/DeepShadingValidationImages/Sibenik/Position/0000000001.exr')['default'][:,:,0:3]

input_image = np.dstack([normal_image, position_image])

batch = np.expand_dims(input_image, axis=0)

print('[INFO] Batch Shape', batch.shape)

data = loaded_model.predict(batch, 1)[0]

print('[INFO] Data Shape', data.shape)

# ToDO: save using Matplotlib
#pyexr.write("out.exr", data)
