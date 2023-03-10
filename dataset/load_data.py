import h5py
import numpy as np
import tqdm
import tensorflow as tf


def load_data(dataset_path):
    h5py_file = h5py.File(dataset_path, 'r')

    train_gazes = []
    train_images = []
    train_poses = []

    test_gazes = []
    test_images = []
    test_poses = []

    test_id = "p00"
    for id in tqdm.tqdm(iterable=range(15), desc='Loading data'):
        person_id = f'p{id:02d}'

        if person_id == test_id:
            test_gazes.extend(h5py_file[f'{person_id}']['gazes'])
            test_images.extend(h5py_file[f'{person_id}']['images'])
            test_poses.extend(h5py_file[f'{person_id}']['poses'])
        else:
            train_gazes.extend(h5py_file[f'{person_id}']['gazes'])
            train_images.extend(h5py_file[f'{person_id}']['images'])
            train_poses.extend(h5py_file[f'{person_id}']['poses'])

    return np.array(train_images), np.array(train_poses), np.array(train_gazes), np.array(test_images), \
        np.array(test_poses), np.array(test_gazes)


class MPIISequence(tf.keras.utils.Sequence):
    def __init__(self, images, poses, gazes, batch_size):
        self.images = images
        self.poses = poses
        self.gazes = gazes
        self.batch_size = batch_size
    def __len__(self):
        return int(np.ceil(len(self.images) / float(self.batch_size)))
    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.poses[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_z = self.gazes[idx * self.batch_size:(idx + 1) * self.batch_size]

        return [batch_x, batch_y], batch_z
    def on_epoch_end(self):
        pass
