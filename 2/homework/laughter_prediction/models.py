import torch
import torch.nn as nn
import torch.nn.functional as F


class LaughterLSTM(nn.Module):

    def __init__(self, input_size, hidden_dim, device):
        super(LaughterLSTM, self).__init__()
        self.device = device

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True)
        self.hidden2class = nn.Linear(hidden_dim, 2)
        self.hidden = None

        self.to(device)

    def _init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(1, batch_size, self.hidden_dim).to(self.device))

    def forward(self, X):
        self.hidden = self._init_hidden(X.size(0))

        outputs, _ = self.lstm(X, self.hidden)
        class_scores = self.hidden2class(outputs)
        class_probs = F.log_softmax(class_scores, dim=-1)

        return class_probs
