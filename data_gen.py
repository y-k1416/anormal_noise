# import library
import glob
import argparse
import sys
import os

import numpy as np
import librosa
import librosa.core
import librosa.feature
import yaml
from tqdm import tqdm 




# load yml
def yaml_load():
    with open("./param.yaml") as stream:
        param = yaml.safe_load(stream)
    return param


# wav file Input
def file_load(file, mono=False):
    """
    load .wav file.

    file : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:

        return librosa.load(file, sr=None, mono=mono)
    except:
        print("file_broken or not exists!! : {}".format(file))


def file_to_vect(file_name, n_mels=64, frames=3, n_fft=1024, hop_length=512, power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa
    y, sr = file_load(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return np.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = np.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T


    return vector_array


def list_to_array(file, msg='data_generate', n_mels=64, frames=3, n_fft=1024, hop_length=512, power=2.0):
    """
    make array
    --------------------
    file : list [ str ]
        dataname
    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * frames
    # iterate file_to_vector_array()
    for idx in range(len(file)):        
        vector_array = file_to_vect(file[idx], n_mels=n_mels, frames=frames, n_fft=n_fft, hop_length=hop_length, power=power)
        if idx == 0:
            dataset = np.zeros((vector_array.shape[0] * len(file), dims), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset

if __name__ in '__main__':
    # make dataset
    training_list_path = os.path.abspath('./data/train/*.wav')
    #training_list_path = './data/train/'
    files = sorted(glob.glob(training_list_path))
    print("============== DATA_GENARATE ==============")
    dataset = list_to_array(files)
    np.save('./sound_data.npy', dataset)