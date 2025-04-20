# generic binary classifier MLP
# this should be the worst performing one: it is the simplest so I'm trying to use this as a baseline
# i.e. is the extra work worth it?

import torch
import torch.nn as nn
import torch.optim as optim
from feature_gen import generate_features
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from torcheval.metrics import MulticlassConfusionMatrix

IN_CHANNELS = 5

class MLP(nn.Module):
    def __init__(self, in_channels = IN_CHANNELS):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(in_channels, 60)
        self.l2 = nn.Linear(60,60)
        self.l3 = nn.Linear(60,30)
        self.l4 = nn.Linear(30, 10)
        self.l5 = nn.Linear(10,1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        z = self.relu(self.l1(input))
        z = self.relu(self.l2(z))
        z = self.relu(self.l3(z))
        z = self.relu(self.l4(z))
        z = self.sigmoid(self.l5(z))
        return z
        
'''
This should also be bad, since MLP can't learn temporal dependencies
- Can only take in flattened vector
'''
class MLPFromSlidingWindow(nn.Module):
    def __init__(self, window_size, num_features):
        super(MLPFromSlidingWindow, self).__init__()
        in_channels = window_size * num_features

        self.relu = nn.ReLU()
        self.l1 = nn.Linear(in_channels, 60)
        # self.l2 = nn.Linear(60, 60)
        self.l3 = nn.Linear(60, 30)
        self.l4 = nn.Linear(30, 10)
        self.l5 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x should be of shape (B, window_size, num_features)
        x = x.view(x.size(0), -1)  # flatten to (B, window_size * num_features)
        z = self.relu(self.l1(x))
        z = self.relu(self.l2(z))
        z = self.relu(self.l3(z))
        z = self.relu(self.l4(z))
        z = self.sigmoid(self.l5(z))
        return z

if __name__ == "__main__":
    mlp = MLP()
    
    # for name, param in cnnlstm.named_parameters():
    #     print(name, param.data)

    # print(cnnlstm.state_dict())
    test_accuracies, test_losses, accuracies, train_losses = [],[],[], []

    from torchsummary import summary
    summary(mlp, (IN_CHANNELS,))

    import os
    print(os.listdir())
    input("OK")

    criterion = nn.BCELoss()  # binary cross-entropy
    optimizer = optim.Adam(mlp.parameters(), lr=1e-2)

    X_np, y_np, feature_names = generate_features(k=IN_CHANNELS)
    X_tensor = torch.from_numpy(X_np)
    y_tensor = torch.from_numpy(y_np)

    print(torch.bincount(y_tensor.squeeze().long()))

    # print(y_tensor)

    train_X = X_tensor
    train_y = y_tensor

    full_dataset = TensorDataset(train_X,train_y)
    # holdout, since kfold doesn't work here
    holdout_size = int(0.2 * len(full_dataset)) # test holdout
    train_size = len(full_dataset) - holdout_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, holdout_size])

    # loaders
    BATCH_SIZE = 16
    # drop last because if not perfect divisibility improperly mismatched sample split will result in crash
    # drop last will mean lost in data so we need a much larger range of dates
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last= True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Training loop: 50 - 100
    for epoch in range(100):
        mlp.train()
        running_loss = 0
        for batch_X, batch_y in train_loader:
            output = mlp(batch_X).squeeze(1)
            loss = criterion(output, batch_y.squeeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        # Evaluation phase
        mlp.eval()
        test_loss = 0
        correct = 0
        total = 0

        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                output = mlp(batch_X).squeeze(1)
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

    mlp.eval()  # important: disables dropout and batchnorm updates
    test_loss = 0
    correct = 0
    total = 0
    conf_matrix = MulticlassConfusionMatrix(num_classes=2)

    with torch.no_grad():  # disables autograd to save memory + speed up
        for batch_X, batch_y in test_loader:
            output = mlp(batch_X).squeeze(1)  # shape: (batch,)

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
    
    import numpy as np
    all_preds_tensor = np.array(all_preds, dtype=np.int64)
    all_targets_tensor = np.array(all_targets, dtype=np.int64)

    # CONFUSION MATRIX FUCKED
    conf_matrix.update(torch.LongTensor(all_preds_tensor), torch.LongTensor(all_targets_tensor))
    print(conf_matrix.compute())

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
    # plt.show()
    plt.savefig(os.path.join("./Models/Plots", "acc_mlp.png"))

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