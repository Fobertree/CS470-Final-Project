import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM:
    def __init__(self, shape):
        self.shape = shape
        self.conv1 = nn.Conv1d(in_channels = 26, out_channels= 64, kernel_size= 3)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(64) 
        # self.avg_pool = nn.AvgPool1d()

        self.lstm1 = nn.LSTM(64, 512, 1, True)

    def forward(self, input):
        z = self.conv1(input)
        z = F.batch_norm(z)
        pass

    def init_weights(self):
        # borrowed this code from CS 334
        for conv in [self.conv1]:
            # since our CNN layer is using ReLU activation, choose Kaiming initialization over Glorot/Xavier initialization
            nn.init.kaiming_uniform_(conv.weight, mode='fan_in', nonlinearity='relu')
            # zero init bias
            nn.init.constant_(conv.bias, 0.0)
        
        # todo: init weight for lstm
        for lstm in [self.lstm1]:
            # xavier bc lstm uses tanh activation
            nn.init.xavier_uniform_(self.lstm.weight_ih, gain=1)
            nn.init.xavier_uniform_(self.lstm.weight_hh, gain=1)
            nn.init.xavier_uniform_(self.fc.weight, gain=1)

            # bias to 0.0
            nn.init.constant_(self.lstm.bias_ih, 0.0)
            nn.init.constant_(self.lstm.bias_hh, 0.0)
            nn.init.constant_(self.fc.bias, 0.0)
