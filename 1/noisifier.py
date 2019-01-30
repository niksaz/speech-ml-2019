# Author: Mikita Sazanovich

import os
import librosa
import random
import numpy as np
import argparse


def load_audio_sec(file_path, sr=16000):
    data, _ = librosa.core.load(file_path, sr)
    return data


def get_random_file_in(directory):
    assert os.path.isdir(directory)
    files = os.listdir(directory)
    file_name = files[random.randrange(len(files))]
    return os.path.join(directory, file_name)


def repeat_or_trim_to(data, wanted_length):
    N = len(data)
    if N > wanted_length:
        return data[:wanted_length]
    else:
        return np.pad(data, (0, wanted_length - N), 'wrap')


def augment_speech(speech_file, output_file, beep_file, back_file, sr):
    speech_data = load_audio_sec(speech_file, sr=sr)
    beep_data = load_audio_sec(beep_file, sr=sr)
    back_data = load_audio_sec(back_file, sr=sr)

    N = len(speech_data)
    beep_data = repeat_or_trim_to(beep_data, N)
    back_data = repeat_or_trim_to(back_data, N)

    assert N == len(beep_data) and N == len(back_data)

    output = speech_data + 0.25 * beep_data + 0.25 * back_data
    librosa.output.write_wav(output_file, output, sr=sr)


def augment_speech_files(speech_dir, output_dir, beep_dir, back_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for speech_filename in os.listdir(speech_dir):
        speech_file = os.path.join(speech_dir, speech_filename)
        speech_base = os.path.splitext(speech_filename)[0]
        output_file = os.path.join(output_dir, speech_base + '-aug.wav')
        beep_file = get_random_file_in(beep_dir)
        back_file = get_random_file_in(back_dir)
        augment_speech(speech_file, output_file, beep_file, back_file, sr=16000)


def main():
    parser = argparse.ArgumentParser(
        description='Augment speech with background and beep noises.')
    parser.add_argument('speech_dir', type=str,
                        help='a path to the directory with speech files')
    parser.add_argument('output_dir', type=str,
                        help='a path to the directory to write the result to')
    parser.add_argument('--beep_dir', type=str, default='noises/beeps',
                        help='a path to the directory with beeps')
    parser.add_argument('--back_dir', type=str, default='noises/background',
                        help='a path to the directory with background noises')
    args = parser.parse_args()

    augment_speech_files(
        args.speech_dir, args.output_dir, args.beep_dir, args.back_dir)


if __name__ == '__main__':
    main()
