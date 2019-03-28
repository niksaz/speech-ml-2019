import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from laughter_prediction.models import LaughterLSTM
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler


class Predictor:
    """
    Wrapper class used for loading serialized model and
    using it in classification task.
    Defines unified interface for all inherited predictors.
    """

    def predict(self, X):
        """
        Predict target values of X given a model

        :param X: numpy.ndarray, dtype=float, shape=[n_samples, n_features]
        :return: numpy.array predicted classes
        """
        raise NotImplementedError("Should have implemented this")

    def predict_proba(self, X):
        """
        Predict probabilities of target class

        :param X: numpy.ndarray, dtype=float, shape=[n_samples, n_features]
        :return: numpy.array target class probabilities
        """
        raise NotImplementedError("Should have implemented this")


class XgboostPredictor(Predictor):
    """Parametrized wrapper for xgboost-based predictors"""

    def __init__(self, model_path, threshold, scaler=None):
        self.threshold = threshold
        self.clf = joblib.load(model_path)
        self.scaler = scaler

    def _simple_smooth(self, data, n=50):
        dlen = len(data)

        def low_pass(data, i, n):
            if i < n // 2:
                return data[:i]
            if i >= dlen - n // 2 - 1:
                return data[i:]
            return data[i - n // 2: i + n - n // 2]

        sliced = np.array([low_pass(data, i, n) for i in range(dlen)])
        sumz = np.array([np.sum(x) for x in sliced])
        return sumz / n

    def predict(self, X):
        y_pred = self.clf.predict_proba(X)
        ypreds_bin = np.where(y_pred[:, 1] >= self.threshold, np.ones(len(y_pred)), np.zeros(len(y_pred)))
        return ypreds_bin

    def predict_proba(self, X):
        X_scaled = self.scaler.fit_transform(X) if self.scaler is not None else X
        not_smooth = self.clf.predict_proba(X_scaled)[:, 1]
        return self._simple_smooth(not_smooth)


class StrictLargeXgboostPredictor(XgboostPredictor):
    """
    Predictor trained on 3kk training examples, using PyAAExtractor
    for input features
    """
    def __init__(self, threshold=0.045985743):
        XgboostPredictor.__init__(self, model_path="models/XGBClassifier_3kk_pyAA10.pkl",
                                  threshold=threshold, scaler=StandardScaler())


class RnnPredictor(Predictor):

    def __init__(self, num_features, hidden_dim, device):
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.device = device
        self.laugh_lstm = LaughterLSTM(num_features, hidden_dim, device)

    def train(self, X_train, y_train, X_test, y_test, epochs=1000, fun_compute_stats=None):
        CLASS_BATCH_SIZE = 1024

        model = self.laugh_lstm
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(self.laugh_lstm.parameters(), lr=1e-1)

        y_train_flat = y_train.flatten()
        non_laughter_ids = np.flatnonzero(y_train_flat == 0)
        laughter_ids = np.flatnonzero(y_train_flat == 1)

        train_losses = []
        test_stats = []
        for _ in tqdm(range(epochs)):
            model.train()
            model.zero_grad()
            class_scores = model(torch.tensor(X_train, dtype=torch.float).to(self.device)).cpu()

            non_laugh_ids = np.random.choice(non_laughter_ids, 2*CLASS_BATCH_SIZE)
            laugh_ids = np.random.choice(laughter_ids, CLASS_BATCH_SIZE)
            ids = np.hstack((non_laugh_ids, laugh_ids))
            targets = torch.tensor(np.take(y_train_flat, ids), dtype=torch.long)
            ids_2d = [id_2d for ind in ids for id_2d in [2 * ind, 2 * ind + 1]]
            class_scores = class_scores.take(torch.tensor(ids_2d)).reshape(-1, 2)

            loss = criterion(class_scores, targets)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            train_losses.append(train_loss)

            if fun_compute_stats is not None:
                model.eval()
                class_scores = model(torch.tensor(X_test, dtype=torch.float).to(self.device)).cpu()
                classes = class_scores.argmax(dim=2).detach()
                y = y_test.reshape(-1)
                y_pred = classes.numpy().reshape(-1)
                test_stats.append(fun_compute_stats(y, y_pred))
        model.eval()
        if fun_compute_stats is None:
            return train_losses
        else:
            return train_losses, test_stats

    def predict(self, X):
        class_scores = self.laugh_lstm(torch.tensor([X], dtype=torch.float).to(self.device)).cpu()
        values = class_scores.argmax(dim=2).detach().numpy()[0]
        return values

    def predict_proba(self, X):
        class_scores = self.laugh_lstm(torch.tensor([X], dtype=torch.float).to(self.device)).cpu()
        probabilities = class_scores.numpy()
        return probabilities
