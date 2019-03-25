import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SpeakerIdentityLSTM(nn.Module):

    def __init__(self, feature_dim, hidden_dim, num_classes, device):
        super(SpeakerIdentityLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device

        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers=1)

        self.hidden2class = nn.Linear(hidden_dim, num_classes)
        self.hidden = None
        self.softmax = nn.LogSoftmax(dim=1)

        self.to(device)

    def __init_hidden(self, batch_size):
        # The axes semantics are (num_layers, batch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(1, batch_size, self.hidden_dim).to(self.device))

    def __get_class_scores(self, batch, lengths):
        self.hidden = self.__init_hidden(batch.size(0))
        packed_input = pack_padded_sequence(batch, lengths, batch_first=True)
        outputs, (ht, ct) = self.lstm(packed_input, self.hidden)
        class_scores = self.hidden2class(ht[-1])

        return class_scores

    def forward(self, batch, lengths):
        class_scores = self.__get_class_scores(batch, lengths)
        output = self.softmax(class_scores)

        return output

    def predict_classes(self, batch, lengths):
        class_scores = self.__get_class_scores(batch, lengths)
        classes = class_scores.argmax(dim=1)

        return classes


class Autoencoder(nn.Module):

    def __init__(self, hidden_dim=13):
        super(Autoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(13, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_dim),
            nn.Tanh())
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 13))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
