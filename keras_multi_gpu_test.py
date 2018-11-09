import tensorflow as tf
from keras.applications import ResNet50
from keras.utils import multi_gpu_model
import numpy as np
num_samples = 1000
height = 32
width = 32
num_classes = 1000
with tf.device('/cpu:0'):
    model = ResNet50(weights=None,
      input_shape=(3, height, width),
      classes=num_classes)
parallel_model = multi_gpu_model(model)
parallel_model.compile(loss='categorical_crossentropy',
    optimizer='rmsprop')
x = np.random.random((num_samples, 3, height, width))
y = np.random.random((num_samples, num_classes))
parallel_model.fit(x, y, epochs=20, batch_size=256)
