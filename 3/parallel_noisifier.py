import os
import subprocess
import multiprocessing

from collections import deque


def main():
    wav_dir = os.path.join('VCTK-Corpus', 'wav48')
    noised_wav_dir = os.path.join('VCTK-Corpus', 'noised')
    os.makedirs(noised_wav_dir, exist_ok=True)
    persons = sorted(os.listdir(wav_dir))
    running_processes = deque()
    for person in persons:
        person_wav_dir = os.path.join(wav_dir, person)
        person_noised_dir = os.path.join(noised_wav_dir, person)
        if os.path.exists(person_noised_dir):
            continue
        cmd = f'python noisifier.py {person_wav_dir} {person_noised_dir}'
        if len(running_processes) == multiprocessing.cpu_count():
            ps = running_processes.pop()
            ps.wait()
        ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        running_processes.append(ps)
    while len(running_processes) > 0:
        ps = running_processes.pop()
        ps.wait()


if __name__ == '__main__':
    main()
