import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm_notebook as tqdm

from classification.models import SpeakerIdentityLSTM, Autoencoder


def _fix_random_state(seed=7):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)


def pack_feature_class_lists(features, classes, ids, device):
    ids = sorted(ids, key=lambda ind: features[ind].size(0), reverse=True)
    lengths = list(map(lambda ind: features[ind].size(0), ids))
    seq = list(map(lambda ind: features[ind], ids))
    inputs = pad_sequence(seq, batch_first=True).to(device)
    targets = torch.tensor(list(map(lambda ind: classes[ind], ids)), dtype=torch.long)
    return inputs, lengths, targets


def train_lstm(train_features, train_classes, test_features, test_classes, num_people, device):
    EPOCHS = 5000
    BATCH_SIZE = 1024
    HIDDEN_DIM = 64
    _fix_random_state()

    model = SpeakerIdentityLSTM(train_features[0].shape[1], HIDDEN_DIM, num_people, device)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)

    N_train = len(train_features)
    N_test = len(test_features)
    train_ids = list(range(N_train))
    test_ids = list(range(N_test))

    train_losses = []
    test_accuracies = []

    for _ in tqdm(range(EPOCHS)):
        np.random.shuffle(train_ids)

        model.train()
        train_loss = 0
        for i in range(0, N_train, BATCH_SIZE):
            model.zero_grad()
            train_inputs, train_lens, train_targets = pack_feature_class_lists(
                train_features, train_classes, train_ids[i: i +BATCH_SIZE], device)
            class_log_probs = model(train_inputs, train_lens).cpu()
            loss = criterion(class_log_probs, train_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss)

        model.eval()
        correct = 0
        for i in range(0, N_test, BATCH_SIZE):
            test_inputs, test_lens, test_targets = pack_feature_class_lists(
                test_features, test_classes ,test_ids[i: i +BATCH_SIZE], device)
            test_predictions = model.predict_classes(test_inputs, test_lens).cpu()
            correct += (test_predictions == test_targets).sum().item()
        accuracy = 100 * correct / N_test
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
    model.eval()
    return model, train_losses
