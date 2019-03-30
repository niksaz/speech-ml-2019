import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import accuracy_score

from classification.models import SpeakerIdentityLSTM, Autoencoder


def _fix_random_state(seed=7):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)


class SoundSegmentsDataset(Dataset):

    def __init__(self, X_features, Y_classes):
        self.X_features = X_features
        self.Y_classes = Y_classes
        assert len(X_features) == len(Y_classes)

    def __len__(self):
        return len(self.X_features)

    def __getitem__(self, idx):
        return self.X_features[idx], self.Y_classes[idx]


def merge_sound_segments_batch(sample):
    N = len(sample)
    ids = sorted(range(N), key=lambda ind: sample[ind][0].size(0), reverse=True)
    lengths = list(map(lambda ind: sample[ind][0].size(0), ids))
    seq = list(map(lambda ind: sample[ind][0], ids))
    inputs = pad_sequence(seq, batch_first=True)
    targets = torch.tensor(list(map(lambda ind: sample[ind][1], ids)), dtype=torch.long)
    return inputs, lengths, targets


def train_lstm(train_dataset, test_dataset, num_people, device):
    EPOCHS = 5000
    BATCH_SIZE = 1024
    HIDDEN_DIM = 64
    _fix_random_state()

    model = SpeakerIdentityLSTM(train_dataset[0][0].shape[1], HIDDEN_DIM, num_people, device)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  collate_fn=merge_sound_segments_batch)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 collate_fn=merge_sound_segments_batch)

    train_losses = []
    test_accuracies = []

    for _ in tqdm(range(EPOCHS)):
        model.train()
        train_loss = 0
        for sample_batched in train_dataloader:
            model.zero_grad()
            train_inputs, train_lens, train_targets = sample_batched
            class_log_probs = model(train_inputs.to(device), train_lens).cpu()
            loss = criterion(class_log_probs, train_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss)

        model.eval()
        predicted_classes = []
        true_classes = []
        for sample_batched in test_dataloader:
            test_inputs, test_lens, test_targets = sample_batched
            test_predictions = model.predict_classes(test_inputs.to(device), test_lens).cpu()
            predicted_classes.extend(test_predictions)
            true_classes.extend(test_targets)
        accuracy = accuracy_score(true_classes, predicted_classes)
        test_accuracies.append(accuracy)
    return train_losses, test_accuracies


def train_autoencoder(noised_features, features, device):
    EPOCHS = 100
    BATCH_SIZE = 1024
    _fix_random_state()

    model = Autoencoder().to(device)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=1e-9)

    xs = []
    ys = []
    for noised_feature, feature in zip(noised_features, features):
        for i in range(noised_feature.shape[0]):
            xs.append(noised_feature[i].numpy())
            ys.append(feature[i].numpy())

    N_train = len(xs)
    print('Total pairs for denoising:', N_train)
    train_ids = list(range(N_train))

    train_losses = []
    for _ in tqdm(range(EPOCHS)):
        np.random.shuffle(train_ids)

        model.train()
        train_loss = 0.0
        for i in range(0, N_train, BATCH_SIZE):
            model.zero_grad()
            ids = train_ids[i:i + BATCH_SIZE]
            seq = list(map(lambda ind: xs[ind], ids))
            X_batch = torch.tensor(seq, dtype=torch.float).to(device)
            Y_pred = model(X_batch).cpu()
            seq = list(map(lambda ind: ys[ind], ids))
            Y_batch = torch.tensor(seq, dtype=torch.float)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss)
    return model, train_losses
