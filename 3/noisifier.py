# Author: Mikita Sazanovich

from pathlib import Path
import librosa
import random
import numpy as np
import argparse
import soundfile

BEEP_SOUND_AMPL_COEF = 0.25
BACK_SOUND_AMPL_COEF = 0.25


def load_audio_sec(file_path, sr=16000):
    data, _ = librosa.core.load(str(file_path), sr)
    return data


def get_random_filepath_in(directory):
    assert directory.is_dir()
    files = list(directory.iterdir())
    filepath = files[random.randrange(len(files))]
    return filepath


def repeat_or_trim_to(data, wanted_length):
    N = len(data)
    if N > wanted_length:
        return data[:wanted_length]
    else:
        return np.pad(data, (0, wanted_length - N), 'wrap')


def augment_speech(speech_path, output_path, beep_path, back_path, sr):
    speech_data = load_audio_sec(speech_path, sr=sr)
    beep_data = load_audio_sec(beep_path, sr=sr)
    back_data = load_audio_sec(back_path, sr=sr)

    N = len(speech_data)
    beep_data = repeat_or_trim_to(beep_data, N)
    back_data = repeat_or_trim_to(back_data, N)

    assert N == len(beep_data) and N == len(back_data)

    output = (speech_data
              + BEEP_SOUND_AMPL_COEF * beep_data
              + BACK_SOUND_AMPL_COEF * back_data)
    soundfile.write(str(output_path), output, samplerate=sr)


def augment_speech_files(speech_dir, output_dir, beep_dir, back_dir):
    print(speech_dir)
    print(output_dir)
    output_dir.mkdir(exist_ok=True)
    for speech_path in speech_dir.iterdir():
        output_path = output_dir / f'{speech_path.stem}.wav'
        beep_path = get_random_filepath_in(beep_dir)
        back_path = get_random_filepath_in(back_dir)
        augment_speech(speech_path, output_path, beep_path, back_path, sr=48000)


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
        Path(args.speech_dir), Path(args.output_dir), Path(args.beep_dir), Path(args.back_dir))


if __name__ == '__main__':
    main()
