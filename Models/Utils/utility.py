import itertools, os
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import numpy as np
import torch.optim as optim
import torch.utils.data as data

# cs 334 boilerplate

def restore_checkpoint(model, checkpoint_dir, cuda=False, force = False):
    """
    If a checkpoint exists, restores the PyTorch model from the checkpoint.
    Returns the model and the current epoch.
    """
    cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
        if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

    if not cp_files:
        print('No saved model parameters found')
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0, []
    
    # Find latest epoch
    for i in itertools.count(1):
        if 'epoch={}.checkpoint.pth.tar'.format(i) in cp_files:
            epoch = i
        else:
            break

    print("Which epoch to load from? Choose in range [1, {}].".format(epoch))
    inp_epoch = int(input())
    if inp_epoch not in range(1, epoch+1):
        raise Exception("Invalid epoch number")

    filename = os.path.join(checkpoint_dir,
        'epoch={}.checkpoint.pth.tar'.format(inp_epoch))

    print("Loading from checkpoint {}".format(filename))
    
    if cuda:
        checkpoint = torch.load(filename, weights_only=False)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filename, weights_only=False,
            map_location=lambda storage, loc: storage)

    try:
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Successfully restored checkpoint (trained for {} epochs)"
            .format(checkpoint['epoch']))
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch

def _evaluate_epoch(data_loader, model, criterion):
    model.eval()
    with torch.no_grad():
        y_true, y_score = [], []
        running_loss = []
        for X, y in data_loader:
            output = model(X)
            y_true.append(y.numpy())
            y_score.append(output)
            running_loss.append(criterion(output, y).item())
        y_true, y_score = np.concatenate(y_true), np.concatenate(y_score)
    
    loss = np.mean(running_loss)
    score = metrics.roc_auc_score(y_true, y_score)
    return loss, score


# model = AirModel()
# optimizer = optim.Adam(model.parameters())
# loss_fn = nn.MSELoss()
# loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
 
# n_epochs = 2000
# for epoch in range(n_epochs):
#     model.train()
#     for X_batch, y_batch in loader:
#         y_pred = model(X_batch)
#         loss = loss_fn(y_pred, y_batch)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # Validation
#     if epoch % 100 != 0:
#         continue
#     model.eval()
#     with torch.no_grad():
#         y_pred = model(X_train)
#         train_rmse = np.sqrt(loss_fn(y_pred, y_train))
#         y_pred = model(X_test)
#         test_rmse = np.sqrt(loss_fn(y_pred, y_test))
#     print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))