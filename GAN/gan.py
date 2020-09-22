import numpy as np
import tensorflow as tf
from tensorflow import keras

coding_size = 30

generator = keras.models.Sequential([
    keras.layers.Dense(100, activation='selu', input_shape=[30]),
    keras.layers.Dense(150, activation='selu'),
    keras.layers.Dense(28*28, activation='sigmoid'),
    keras.layers.Reshape([28, 28])
])

discriminator = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(150, activation='selu'),
    keras.layers.Dense(100, activation='selu'),
    keras.layers.Dense(1, activation='sigmoid')
])

gan = keras.models.Sequential([generator, discriminator])

discriminator.compile(loss=keras.losses.binary_crossentropy, optimizer='rmsprop')
discriminator.trainable = False

gan.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.RMSprop())

# Loading Data
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

batch_size = 32

dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset = dataset.batch(batch_size).prefetch(1)

def train_gan(gan, dataset, batch_size, coding_size, n_epochs=50, n_discriminator_train_step=5):
    generator, discriminator = gan.layers
    for epoch in n_epochs:
        for X_batch in X_train:
            noise = tf.random.normal([batch_size, coding_size])
            fake_X = generator(noise)
            # real followed by fake
            X_real_and_fake = tf.concat([X_batch, fake_X])
            y1 = tf.constant([[1.]] * batch_size + [[0.]] * batch_size)
            discriminator.trainable = True
            for _ in range(n_discriminator_train_step):
                discriminator.train_on_batch(X_real_and_fake, y1)
            
            # update generator
            discriminator.trainable=False
            noise = tf.random.normal([batch_size, coding_size])
            y2 = tf.constant([[1.]] * batch_size)
            gan.train_on_batch(noise, y2)

train_gan(gan, dataset, batch_size, coding_size)
