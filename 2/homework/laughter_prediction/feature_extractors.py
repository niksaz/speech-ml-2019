import os
import tempfile
import librosa
import numpy as np
from scipy.io import wavfile
import pandas as pd


class FeatureExtractor:
    def extract_features(self, wav_path):
        """
        Extracts features for classification ny frames for .wav file

        :param wav_path: string, path to .wav file
        :return: pandas.DataFrame with features of shape (n_chunks, n_features)
        """
        raise NotImplementedError("Should have implemented this")


class PyAAExtractor(FeatureExtractor):
    """Python Audio Analysis features extractor"""
    def __init__(self):
        self.extract_script = "./extract_pyAA_features.py"
        self.py_env_name = "ipykernel_py2"

    def extract_features(self, wav_path):
        with tempfile.NamedTemporaryFile() as tmp_file:
            feature_save_path = tmp_file.name
            cmd = "python \"{}\" --wav_path=\"{}\" " \
                  "--feature_save_path=\"{}\"".format(self.extract_script, wav_path, feature_save_path)
            os.system("source activate {}; {}".format(self.py_env_name, cmd))

            feature_df = pd.read_csv(feature_save_path)
        return feature_df


class LibrosaExtractor(FeatureExtractor):
    def __init__(self, frame_sec=0.5):
        self.frame_sec = frame_sec

    def extract_features(self, wav_path):
        rate, audio = wavfile.read(wav_path)
        audio = audio.astype(np.float64)
        N = audio.shape[0]
        frame_length = int(rate * self.frame_sec)
        print(frame_length)
        frame_shift = frame_length // 4
        features = []
        for i in range(0, N - frame_length + 1, frame_shift):
            features.append(np.concatenate((
                np.mean(librosa.feature.mfcc(audio[i: i + frame_length], rate).T, axis=0),
                np.mean(librosa.feature.melspectrogram(audio[i: i + frame_length], rate).T, axis=0)
            )))
        return pd.DataFrame(np.array(features))
