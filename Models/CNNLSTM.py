import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

class CNNLSTM(nn.Module):
    def __init__(self, in_channels = 26):
        super(CNNLSTM, self).__init__()

        # input tensor: 26 x 1?
        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels= 64, kernel_size= 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.batch_norm1 = nn.BatchNorm1d(64)
        self.avg_pool = nn.AvgPool1d(64)
        # if we have time, we can separate dropouts and treat droppout rate as hyperparameter
        self.dropout = nn.Dropout1d()

        # 4x because 4 gates: input gate, output gate, cell gate, forget gate
        self.lstm1 = nn.LSTM(64, 128, 1, True)
        self.batch_norm2 = nn.BatchNorm1d(128)

        self.lstm2 = nn.LSTM(128, 80, 1, True)
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

        # permute used to reshape before LSTM (reorder)
        # [batch_size, feature_size, sequence_length] -> [batch_size, sequence_length, feature_size]
        z = z.permute(0, 2, 1)
        z, _ = self.lstm1(z)
        z = self.tanh(z)

        z = z[:, -1, :]
        z = self.batch_norm2(z)
        z = self.dropout(z)

        z = z.unsqueeze(1)
        z, _ = self.lstm2(z)
        z = self.tanh(z)

        z = z[:, -1, :]
        z = self.batch_norm3(z)
        z = self.dropout(z)
        
        z = self.dense(z)
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
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0.0)

if __name__ == "__main__":
    cnnlstm = CNNLSTM()
    
    # for name, param in cnnlstm.named_parameters():
    #     print(name, param.data)

    # print(cnnlstm.state_dict())

    from torchsummary import summary
    summary(cnnlstm, (26, 100))

    # input("Ready to train")

    # TRAINING BOILERPLATE BELOW
    model = CNNLSTM(in_channels=26)

    criterion = nn.BCELoss()  # binary cross-entropy
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # NOT RIGHT SHAPE
    X = torch.randn(16, 26, 100)         # (B, C, T) format for Conv1D
    y = torch.randint(0, 2, (16, 1)).float()  # binary labels as floats

    # Create DataLoader
    full_dataset = TensorDataset(X, y)

    # holdout, since kfold doesn't work here
    holdout_size = int(0.2 * len(full_dataset)) # test holdout
    train_size = len(full_dataset) - holdout_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, holdout_size])
    
    # loaders
    BATCH_SIZE = 3
    # drop last because if not perfect divisibility improperly mismatched sample split will result in crash
    # drop last will mean lost in data so we need a much larger range of dates
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last= True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Training loop
    for epoch in range(20):
        model.train()
        for batch_X, batch_y in train_loader:
            # Forward pass
            output = model(batch_X).squeeze(1)  # shape: (batch,)
            loss = criterion(output, batch_y.squeeze(1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Test loop

    model.eval()  # important: disables dropout and batchnorm updates
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # disables autograd to save memory + speed up
        for batch_X, batch_y in test_loader:
            output = model(batch_X).squeeze(1)  # shape: (batch,)

            # BCELoss            
            loss = criterion(output, batch_y.squeeze(1))
            test_loss += loss.item()

            # convert probabilities to binary predictions
            preds = (output >= 0.5).float()

            correct += (preds == batch_y.squeeze(1)).sum().item()
            total += batch_y.size(0)

    # WARNING: if batch size too high, len(test_loader) will return 0
    avg_test_loss = test_loss / len(test_loader)
    accuracy = correct / total

    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2%}")