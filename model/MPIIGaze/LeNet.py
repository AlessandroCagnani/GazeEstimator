import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Dense, Concatenate, Flatten, ReLU


class LeNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(filters=20, kernel_size=(5, 5), strides=(1, 1),
                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                            bias_initializer='zeros', name='conv1')
        self.max_pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='maxpool1')
        self.conv2 = Conv2D(filters=50, kernel_size=(5, 5), strides=(1, 1),
                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                            bias_initializer='zeros', name='conv2')
        self.max_pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='maxpool2')
        self.flatten = Flatten(name='flatten')
        self.dense1 = Dense(units=500, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros',
                            name='dense1')
        self.concat = Concatenate(axis=-1, name='concat')
        self.dense2 = Dense(units=2, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='dense2')

    def call(self, inputs):
        input_img, input_pose = inputs
        conv1 = self.conv1(input_img)
        pool1 = self.max_pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.max_pool2(conv2)
        flatt = self.flatten(pool2)
        dense1 = self.dense1(flatt)
        cat = self.concat([dense1, input_pose])
        dense2 = self.dense2(cat)
        return dense2
