# generic binary classifier MLP
# this should be the worst performing one: it is the simplest so I'm trying to use this as a baseline
# i.e. is the extra work worth it?

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_channels = 13):
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
        self.l2 = nn.Linear(60, 60)
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

    from torchsummary import summary
    summary(mlp, (13,))