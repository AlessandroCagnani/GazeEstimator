import h5py
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from dataset.load_data import load_data, MPIISequence
from model import create_model
import os
import sklearn
import datetime

def main():
    model = create_model()


    dataset_path = 'dataset/dataset/processed/MPIIGaze.h5'
    output_dir = 'model/trained/mpiigaze/'

    day_and_time = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-").split(".")[0]
    # Load the data
    train_images, train_poses, train_labels, val_images, val_poses, val_labels = load_data(dataset_path)
    # reshape each image to (36, 60, 1) and convert to float32 and normalize
    train_images = train_images.reshape(train_images.shape[0], 36, 60, 1).astype('float32') / 255
    val_images = val_images.reshape(val_images.shape[0], 36, 60, 1).astype('float32') / 255

    sklearn.utils.shuffle(train_images, train_poses, train_labels)

    sequence = MPIISequence(train_images, train_poses, train_labels, batch_size=1000)

    # print("-------- shapes --------")
    # print("train_images: ", train_images.shape)
    # print("train_poses: ", train_poses.shape)
    # print("train_labels: ", train_labels.shape)
    # print("val_images: ", val_images.shape)
    # print("val_poses: ", val_poses.shape)
    # print("val_labels: ", val_labels.shape)
    # print("--------        --------")

    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
    # model.summary()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    callbacks = [ModelCheckpoint(os.path.join(output_dir, 'MPIIGaze.h5'), verbose=1, save_best_only=True)]

    history = model.fit(sequence,
                        epochs=10,
                        validation_data=([val_images, val_poses], val_labels),
                        batch_size=1000,
                        # callbacks=callbacks,
                        verbose=1)

    score = model.evaluate([val_images, val_poses], val_labels, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    if not os.path.exists(os.path.join(output_dir, f'lenet_{day_and_time}')):
        os.makedirs(os.path.join(output_dir, f'lenet_{day_and_time}'))
    model.save(os.path.join(output_dir, f'lenet_{day_and_time}'), save_format='tf')

    if not os.path.exists(os.path.join(output_dir, 'weights', f'lenet_{day_and_time}')):
        os.makedirs(os.path.join(output_dir, 'weights', f'lenet_{day_and_time}'))
    model.save_weights(os.path.join(output_dir, 'weights', f'lenet_{day_and_time}', 'weights.h5'))


if __name__ == "__main__":
    main()