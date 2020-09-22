import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import activations
from tensorflow.python.keras.layers.core import Reshape
from tensorflow.python.ops.gen_array_ops import pad
from tensorflow.python.ops.gen_math_ops import mod, xdivy

coding_size = 100

generator = keras.models.Sequential([
    keras.layers.Dense(4*4*1024, activation='relu', input_shape=[coding_size]),
    keras.layers.Reshape([4, 4, 1024]),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(512, 5, 2, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(256, 5, 2, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(128, 5, 2, padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(3, 5, 2, padding='same', activation='tanh')

])

model = keras.models.Sequential([
    
])