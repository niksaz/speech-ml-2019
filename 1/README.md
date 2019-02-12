# Data augmentation

You could find example audio files at **./samples**. 


#### HOWTO
Usage is the following:

```
python noisifier.py --help
usage: noisifier.py [-h] [--beep_dir BEEP_DIR] [--back_dir BACK_DIR]
                    speech_dir output_dir

Augment speech with background and beep noises.

positional arguments:
  speech_dir           a path to the directory with speech files
  output_dir           a path to the directory to write the result to

optional arguments:
  -h, --help           show this help message and exit
  --beep_dir BEEP_DIR  a path to the directory with beeps
  --back_dir BACK_DIR  a path to the directory with background noises
  ```