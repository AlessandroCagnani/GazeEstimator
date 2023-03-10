import os
import tqdm
import re
import numpy as np
import scipy.io
import pandas as pd
import cv2
import argparse
import h5py


column_names = ['p, day ,left_image', 'left_pose', 'left_gaze', 'right_image', 'right_pose', 'right_gaze']

def convert_pose(vector) -> np.ndarray:
    rot = cv2.Rodrigues(np.array(vector).astype(np.float32))[0]
    vec = rot[:, 2]
    pitch = np.arcsin(vec[1])
    yaw = np.arctan2(vec[0], vec[2])
    return np.array([pitch, yaw]).astype(np.float32)

def convert_gaze(vector) -> np.ndarray:
    x, y, z = vector
    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)
    return np.array([pitch, yaw]).astype(np.float32)



def mat_reader(mat_path):

    mat_file = scipy.io.loadmat(mat_path, struct_as_record=False ,squeeze_me=True)
    mat_data = mat_file['data']
    num_data = len(mat_file['filenames']) if type(mat_file['filenames']) == np.ndarray else 1
    r_data = mat_data.right
    l_data = mat_data.left

    # if r_data.image.ndim < 3:
    #     print(f'{mat_path} only one img, {r_data.image.shape}')
    #     print(l_data.image)
    images = []
    poses = []
    gazes = []

    assert r_data.image.shape == l_data.image.shape, "Images are not the same size"
    assert (l_data.image.shape[0] == l_data.pose.shape[0] and r_data.pose.shape[0] == r_data.gaze.shape[0])\
            or (r_data.pose.ndim == 1 and r_data.gaze.ndim == 1 and r_data.image.ndim == 2), "Not the same number of images and poses/gazes"

    for idx in range(num_data):
        # left
        if num_data == 1:
            l_img = l_data.image
            l_pose = convert_pose(l_data.pose)
            l_gaze = convert_gaze(l_data.gaze)
            # right
            r_img = r_data.image[:, ::-1]
            r_pose = convert_pose(r_data.pose) * np.array([1, -1])
            r_gaze = convert_gaze(r_data.gaze) * np.array([1, -1])
        else:
            l_img = l_data.image[idx]
            l_pose = convert_pose(l_data.pose[idx])
            l_gaze = convert_gaze(l_data.gaze[idx])
            # right
            r_img = r_data.image[idx][:, ::-1]
            r_pose = convert_pose(r_data.pose[idx]) * np.array([1, -1])
            r_gaze = convert_gaze(r_data.gaze[idx]) * np.array([1, -1])

        images.append(l_img)
        poses.append(l_pose)
        gazes.append(l_gaze)
        images.append(r_img)
        poses.append(r_pose)
        gazes.append(r_gaze)

    return images, poses, gazes



def main():
    output_dir = 'dataset/processed'
    output_file = os.path.join(output_dir, "MPIIGazeTest.h5")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_file):
        print(f"Removing existing file {output_file}, replacing ...")
        os.remove(output_file)
    else:
        print(f"Creating new file {output_file} ...")

    normalized_data = 'dataset/MPIIGaze/Data/Normalized'
    eval_data = 'dataset/MPIIGaze/data/Evaluation/sample list for eye images'
    data = dict(images=list(), poses=list(), gazes=list(), user=list())

    for person_id in tqdm.tqdm(iterable=range(15), desc='Reading data'):
        person_id = f'p{person_id:02d}'
        person_dir = os.path.join(normalized_data, person_id)
        person_images = []
        person_poses = []
        person_gazes = []
        for day_file in sorted(os.listdir(person_dir)):
            images, poses, gazes = mat_reader(os.path.join(person_dir, day_file))
            assert len(images) == len(poses) == len(gazes), 'Data length mismatch'
            person_images.extend(images)
            person_poses.extend(poses)
            person_gazes.extend(gazes)
            day = re.findall(r'\d\d', day_file)[0]
        person_images = np.asarray(person_images).astype(np.uint8)
        person_poses = np.asarray(person_poses).astype(np.float32)
        person_gazes = np.asarray(person_gazes).astype(np.float32)

        # print(person_id)

        with h5py.File(output_file, 'a') as f_output:
            f_output.create_dataset(f'{person_id}/images', data=person_images)
            f_output.create_dataset(f'{person_id}/poses', data=person_poses)
            f_output.create_dataset(f'{person_id}/gazes', data=person_gazes)

if __name__ == '__main__':
    main()



