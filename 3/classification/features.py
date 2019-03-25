import torch

from numpy import mean
from sklearn.preprocessing import scale

from kaldi.feat.mfcc import Mfcc, MfccOptions
from kaldi.matrix import SubVector, SubMatrix
from kaldi.util.table import SequentialWaveReader, SequentialMatrixReader, MatrixWriter


def compute_kaldi_features(desc_path, output_path, samp_freq=8000):
    print('Started processing', desc_path)
    rspec, wspec = 'scp:{}'.format(desc_path), 'ark:{}'.format(output_path)

    mfcc_opts = MfccOptions()
    mfcc_opts.frame_opts.samp_freq = samp_freq

    mfcc = Mfcc(mfcc_opts)
    sf = mfcc_opts.frame_opts.samp_freq

    with SequentialWaveReader(rspec) as reader, MatrixWriter(wspec) as writer:
        for key, wav in reader:
            assert (wav.samp_freq >= sf)
            assert (wav.samp_freq % sf == 0)

            s = wav.data()
            s = s[:, ::int(wav.samp_freq / sf)]
            m = SubVector(mean(s, axis=0))

            f = mfcc.compute_features(m, sf, 1.0)
            f = SubMatrix(scale(f))
            writer[key] = f
    print('Success! Result is written to', output_path)


def write_kaldi_descriptor(wav_paths, desc_path):
    with open(desc_path, 'w') as output:
        for wav_path in wav_paths:
            output.write('{} {}\n'.format(wav_path, wav_path))


def read_kaldi_features_to_tensor(kaldi_path):
    features = []
    rspec = 'ark:{}'.format(kaldi_path)
    with SequentialMatrixReader(rspec) as reader:
        for key, mat in reader:
            features.append(torch.tensor(mat, dtype=torch.float))
    return features


def compute_auto_features(autoencoder, features_all):
    auto_features_all = []
    for features in features_all:
        K = features.shape[0]
        auto_features = []
        for k in range(K):
            auto_features.append(autoencoder.encode(features[k]).detach())
        auto_features_all.append(torch.stack(auto_features))
    return auto_features_all


def build_merged_features(features1_all, features2_all):
    merged_features_all = []
    for features1, features2 in zip(features1_all, features2_all):
        merged_features = torch.cat((features1, features2), dim=1)
        merged_features_all.append(merged_features)
    return merged_features_all
