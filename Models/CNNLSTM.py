import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    def __init__(self, in_channels = 26):
        super(CNNLSTM, self).__init__()

        # input tensor: 26 x 1?
        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels= 64, kernel_size= 3)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.batch_norm1 = nn.BatchNorm2d(64) 
        self.avg_pool = nn.AvgPool1d(64)
        # if we have time, we can separate dropouts and treat droppout rate as hyperparameter
        self.dropout = nn.Dropout1d()

        self.lstm1 = nn.LSTM(64, 512, 1, True)
        self.batch_norm2 = nn.BatchNorm1d(128)

        self.lstm2 = nn.LSTM(128, 320, 1, True)
        self.batch_norm3 = nn.BatchNorm1d(80)
        self.dense = nn.Linear(80, 1) # 1 output: binary

    def forward(self, input):
        # can use nn.Sequential as well
        '''
        https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
        '''
        z = self.conv1(input)
        z = self.relu(z)
        z = self.batch_norm1(z)
        z = self.avg_pool(z)
        z = self.dropout(z)
        z, _ = self.lstm1(z)
        z = self.tanh(z)
        z = self.batch_norm2(z)
        z = self.dropout(z)
        z, _ = self.lstm2(z)
        z = self.tanh(z)
        z = self.batch_norm3(z)
        z = self.dropout(z)
        z - self.dense(z)
        z = self.sigmoid(z)
        return z

    def init_weights(self):
        # borrowed this code from CS 334
        for conv in [self.conv1]:
            # since our CNN layer is using ReLU activation, choose Kaiming initialization over Glorot/Xavier initialization
            nn.init.kaiming_uniform_(conv.weight, mode='fan_in', nonlinearity='relu')
            # zero init bias
            nn.init.constant_(conv.bias, 0.0)
        
        # todo: init weight for lstm
        for lstm in [self.lstm1, self.lstm2]:
            # xavier bc lstm uses tanh activation
            nn.init.xavier_uniform_(lstm.weight_ih, gain=1)
            nn.init.xavier_uniform_(lstm.weight_hh, gain=1)
            nn.init.xavier_uniform_(lstm.fc.weight, gain=1)

            # bias to 0.0
            nn.init.constant_(lstm.bias_ih, 0.0)
            nn.init.constant_(lstm.bias_hh, 0.0)
            nn.init.constant_(lstm.fc.bias, 0.0)

if __name__ == "__main__":
    cnnlstm = CNNLSTM()
    
    for name, param in cnnlstm.named_parameters():
        print(name, param.data)

    print(cnnlstm.state_dict())

    from torchsummary import summary
    summary(cnnlstm, (26,))