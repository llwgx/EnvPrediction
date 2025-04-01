import torch
from torch import nn
from sklearn.svm import SVR
import numpy as np


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, n_steps, n_units, num_layers=3):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, n_units, num_layers, batch_first=True)
        self.fc = nn.Linear(n_units, n_steps)  # 输出维度改为 n_steps

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 取最后一个时间步的 LSTM 输出，并映射到 n_steps 维
        return out


class SimpleLiner(nn.Module):
    def __init__(self, input_dim, n_steps, n_units, num_layers=2):
        super(SimpleLiner, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_units)  # 输出维度改为 n_steps
        self.relu = nn.ReLU()  # 添加ReLU激活函数
        self.fc2 = nn.Linear(n_units, n_steps)  # 输出维度改为 n_steps

    def forward(self, x):
        x = x.squeeze(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimpleGRU(nn.Module):
    def __init__(self, input_dim, n_steps, n_units, num_layers=3):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_dim, n_units, num_layers, batch_first=True)
        self.fc = nn.Linear(n_units, n_steps)  # 输出维度为 n_steps

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])  # 取最后一个时间步的输出
        return out


class SimpleRNN(nn.Module):
    def __init__(self, input_dim, n_steps, n_units, num_layers=3):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, n_units, num_layers, batch_first=True)
        self.fc = nn.Linear(n_units, n_steps)  # 输出维度为 n_steps

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        out = self.fc(rnn_out[:, -1, :])  # 取最后一个时间步的输出
        return out


class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, n_steps, n_units, num_heads=7, num_layers=2):
        super(SimpleTransformer, self).__init__()
        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})")
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, n_steps)

    def forward(self, x):
        x = self.transformer(x)
        x = x.mean(dim=1)  # 取所有时间步的均值
        out = self.fc(x)
        return out


class SVRModel:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.model = SVR(kernel=kernel, C=C, gamma=gamma)

    def fit(self, X, y):
        X = X.reshape(X.shape[0], -1)  # 展平输入数据
        self.model.fit(X, y)

    def predict(self, X):
        X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)