import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from feature_gen import generate_features

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
        # INPUT DIMS: (batch_size, channels, features)
        # [samples/batch size, timesteps, features]
        # channels should be # of features in feature vec
        # features for time series case: look-back window in timeseries?
        # expects [batch size, num features, time]
        # print(input.shape)
        z = self.conv1(input)
        z = self.relu(z)
        z = self.batch_norm1(z)
        z = self.avg_pool(z)
        z = self.dropout(z)

        # permute used to reshape before LSTM (reorder)
        # [batch_size, feature_size, sequence_length] -> [batch_size, sequence_length, feature_size]
        z = z.permute(0, 2, 1)
        # lstm returns output, (hn: final hidden state, low bar, cn: internel cell state, top bar)
        z, _ = self.lstm1(z)
        z = self.tanh(z)

        # get last timestep
        z = z[:, -1, :]
        z = self.batch_norm2(z)
        z = self.dropout(z)

        # add extra dim in 1 in prep for lstm
        z = z.unsqueeze(1)
        z, _ = self.lstm2(z)
        z = self.tanh(z)

        # get last timestep
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

# Utility to create proper 3D tensor for CNNLSTM: time-lagged sliding windows
# Worry: this could maybe leak into testing?
def create_sliding_windows(X, window_size):
    sequences = []
    for i in range(len(X) - window_size + 1):
        seq = X[i:i+window_size]
        sequences.append(seq)
    return torch.stack(sequences)

if __name__ == "__main__":
    train_losses = []
    test_losses = []
    accuracies = []
    cnnlstm = CNNLSTM()
    
    # for name, param in cnnlstm.named_parameters():
    #     print(name, param.data)

    # print(cnnlstm.state_dict())

    # from torchsummary import summary
    # summary(cnnlstm, (26, 100))

    # input("Ready to train")

    # TRAINING BOILERPLATE BELOW
    # model = CNNLSTM(in_channels=26)
    model = CNNLSTM(in_channels=15)  # 15 features, 100 timesteps

    criterion = nn.BCELoss()  # binary cross-entropy
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Must be T (lag timesteps) >= 66 for some reason 
    # X = torch.randn(16, 26, 100)         # (B, C, T) format for Conv1D
    window_size = 100
    # X = torch.randn(363, 26)
    # y = torch.randint(0, 2, (363, 1)).float()  # binary labels as floats
    X_np, y_np, feature_names = generate_features(k=15)
    X_tensor = torch.from_numpy(X_np)
    y_tensor = torch.from_numpy(y_np)

    X = X_tensor
    y = y_tensor

    # Prevent leakage by splitting first BEFORE generating sliding window tensors
    # TimeSeriesSplit? https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
    split_idx = int(0.8 * len(X))
    train_raw = X[:split_idx]
    test_raw = X[split_idx - window_size:]

    train_X = create_sliding_windows(train_raw, window_size).permute(0,2,1)
    test_X = create_sliding_windows(test_raw, window_size).permute(0,2,1)

    train_y = y[:split_idx-window_size+1]
    test_y = y[split_idx - window_size+1:]

    # X = create_sliding_windows(X, window_size).permute(0,2,1) #permute spaghetti here so the cnn can take it in
    # y = torch.randint(0, 2, (363 - window_size + 1, 1)).float()  # binary labels as floats

    # Create DataLoader
    full_dataset = TensorDataset(train_X,train_y)
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

    # Training loop: 50 - 100
    for epoch in range(20):
        model.train()
        running_loss = 0
        for batch_X, batch_y in train_loader:
            output = model(batch_X).squeeze(1)
            loss = criterion(output, batch_y.squeeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        # Evaluation phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                output = model(batch_X).squeeze(1)
                loss = criterion(output, batch_y.squeeze(1))
                test_loss += loss.item()

                probs = output
                preds = (probs >= 0.5).float()

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.squeeze(1).cpu().numpy())

                correct += (preds == batch_y.squeeze(1)).sum().item()
                total += batch_y.size(0)

        avg_test_loss = test_loss / len(test_loader)
        accuracy = correct / total

        test_losses.append(avg_test_loss)
        accuracies.append(accuracy)

        print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2%}")
    
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

    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    auroc = roc_auc_score(all_targets, all_probs)

    print(f"\nEvaluation Metrics:")
    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")

    epochs = range(1, len(train_losses)+1)

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label="Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.show()

    input("OK")

    fpr, tpr, _ = roc_curve(all_targets, all_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUROC = {auroc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()