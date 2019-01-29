# Author: Mikita Sazanovich

import os
import librosa
import random
import numpy as np


def get_random_file_in(direct):
    assert os.path.isdir(direct)
    files = os.listdir(direct)
    file_ind = random.randrange(len(files))
    return files[file_ind]


def load_audio_sec(file_path, sr=16000):
    data, _ = librosa.core.load(file_path, sr)
    return data


def repeat_or_trim_to(data, wanted_length):
    N = len(data)
    if N > wanted_length:
        return data[:wanted_length]
    else:
        return np.pad(data, (0, wanted_length - N), 'wrap')


def main():
    sr = 16000
    speech_file = 'samples/speech.wav'
    speech_data = load_audio_sec(speech_file, sr)

    beep_dir = 'noises/beeps'
    random_beep_file = get_random_file_in(beep_dir)
    beep_data = load_audio_sec(os.path.join(beep_dir, random_beep_file), sr)

    back_dir = 'noises/background'
    random_back_file = get_random_file_in(back_dir)
    back_data = load_audio_sec(os.path.join(back_dir, random_back_file), sr)

    N = len(speech_data)
    beep_data = repeat_or_trim_to(beep_data, N)
    back_data = repeat_or_trim_to(back_data, N)

    assert N == len(beep_data) and N == len(back_data)

    output = speech_data + 0.5 * beep_data + 0.5 * back_data
    librosa.output.write_wav('output/augmented.wav', output, sr=sr)


if __name__ == '__main__':
    main()
