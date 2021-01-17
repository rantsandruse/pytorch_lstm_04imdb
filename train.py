'''
Training and testing
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score


def train_val_test_split(X, X_lens, y, train_val_split = 10, trainval_test_split = 10):
    '''
    Pre-split data to train, val and test.
    Parameters
    ----------
    X
    X_lens
    y
    train_val_split
    trainval_test_split

    Returns
    -------

    '''
    # We switch over to stratified kfold
    splits = StratifiedKFold(n_splits=trainval_test_split, shuffle=True, random_state=42)
    for trainval_idx, test_idx in splits.split(X, y):
        X_trainval, X_test = X[trainval_idx], X[test_idx]
        y_trainval, y_test = y[trainval_idx], y[test_idx]
        X_lens_trainval, X_lens_test = X_lens[trainval_idx], X_lens[test_idx]

    splits = StratifiedKFold(n_splits=train_val_split, shuffle=True, random_state=28)

    for train_idx, val_idx in splits.split(X_trainval, y_trainval):
        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
        X_lens_train, X_lens_val = X_lens_trainval[train_idx], X_lens_trainval[val_idx]

    train_dataset = TensorDataset(torch.tensor(X_train, dtype = torch.long),
                                  torch.tensor(y_train, dtype=torch.long),
                                  torch.tensor(X_lens_train, dtype=torch.int64))

    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.long),
                                torch.tensor(y_val, dtype=torch.long),
                                torch.tensor(X_lens_val, dtype=torch.int64))

    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.long),
                                 torch.tensor(y_test, dtype=torch.long),
                                 torch.tensor(X_lens_test, dtype=torch.int64))


    return train_dataset, val_dataset, test_dataset


# Change No.2: CrossEntropyLoss() --> BCEWithLogitsLoss()
def train(model, train_dataset, val_dataset, test_dataset, loss_fn, optimizer, n_epochs = 5, batch_size = 2, device = "gpu", patience = 3):
    '''
    Parameters
    ----------
    model
    X
    y
    X_lens
    optimizer
    loss_fn
    n_epochs
    batch_size
    seq_len

    Returns
    -------

    '''
    # Use scikit learn stratified k-fold.
    # I gave up on the initial choice of pytorch random_split, as it would not return indices.
    # train_dataset, val_dataset = random_split(dataset, [16,4])
    device = torch.device("cuda" if torch.cuda.is_available() and device=="gpu" else "cpu")

    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size)
    val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size)
    test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size)

    epoch_train_losses = []
    epoch_val_acc = []
    epoch_train_acc = []
    epoch_test_acc = []
    min_val_acc = None
    patience_counter = 0

    for epoch in range(n_epochs):
        model.to(device)
        train_losses, val_losses = [], []

        for X_train, y_train, X_lens_train in train_loader:
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            # We need to preserve X_lens_batch on CPU, as the behavior of as_tensor has changed:
            # And now pack_padded_sequence requires a cpu tensor.
            # IMHO This is more of a bug than a feature.
            # The issue is documented here: https://github.com/pytorch/pytorch/issues/43227
            # X_lens_batch = X_lens_batch.to(device)
            optimizer.zero_grad()
            ypred_train = model(X_train, X_lens_train)
            # Change No.3:
            # The Loss function does not need to be reshaped here, but we need to make sure that
            # input and target dimensions the same.
            # Also BCEwithlogits is peculiar about taking in floats.
            train_loss = loss_fn(ypred_train.float(), y_train.float())
            train_loss.backward()
            train_losses.append(train_loss.item())

            optimizer.step()

        # with torch.no_grad():
        #     for X_val, y_val, X_lens_val in val_loader:
        #         X_val = X_val.to(device)
        #         y_val = y_val.to(device)
        #         # X_lens_val = X_lens_val.to(device)
        #         ypred_val = model(X_val, X_lens_val)
        #         # Change No.3:
        #         # The Loss function does not need to be reshaped here, but we need to make sure that
        #         # input and target dimensions the same.
        #         # Also BCEwithlogits is peculiar about taking in floats.
        #         val_loss = loss_fn(ypred_val.float(), y_val.float())
        #         val_losses.append(val_loss.item())
        #
        #         ypred_val_all.append(torch.sigmoid(ypred_val))
        #         y_val_all.append(y_val)
        #
        #     for X_train, y_train, X_lens_train in val_loader:
        #         X_train = X_train.to(device)
        #         y_train = y_train.to(device)
        #         # X_lens_val = X_lens_val.to(device)
        #         ypred_train = model(X_train, X_lens_train)
        #         # Change No.3:
        #         # The Loss function does not need to be reshaped here, but we need to make sure that
        #         # input and target dimensions the same.
        #         # Also BCEwithlogits is peculiar about taking in floats.
        #         ypred_train_all.append(torch.sigmoid(ypred_train))
        #         y_train_all.append(y_train)
        #
        #     for X_test, y_test, X_lens_test in test_loader:
        #         X_test = X_test.to(device)
        #         y_test = y_test.to(device)
        #         # X_lens_test = X_lens_test.to("cpu")
        #         ypred_test = model(X_test, X_lens_test)
        #         ypred_test_all.append(torch.sigmoid(ypred_test))
        #         y_test_all.append(y_test)
        #
        #
        # curr_val_loss = np.mean(val_losses)
        #
        # ypred_val_all = torch.cat(ypred_val_all)
        # y_val_all = torch.cat(y_val_all)

        curr_train_loss = np.mean(train_losses)
        ypred_val_all, y_val_all = eval(model, val_dataset, batch_size)
        ypred_train_all, y_train_all = eval(model, train_dataset, batch_size)
        ypred_test_all, y_test_all = eval(model, test_dataset, batch_size)

        _, _, curr_val_acc,_ = calc_metrics(ypred_val_all, y_val_all)
        _, _, curr_train_acc, _ = calc_metrics(ypred_train_all, y_train_all)
        _, _, curr_test_acc, _ = calc_metrics(ypred_test_all, y_test_all)

        epoch_train_losses.append(curr_train_loss)
        epoch_val_acc.append(curr_val_acc)
        epoch_train_acc.append(curr_train_acc)
        epoch_test_acc.append(curr_test_acc)

        # implement early stopping.
        # stop when the validation loss shows monotonic increase Patience number of times.
        # Early stopping. Will not apply when patience = -1
        if patience > -1:
            if min_val_acc is None:
                min_val_acc = curr_val_acc
            elif curr_val_acc <= min_val_acc:
                min_val_loss = curr_val_acc
                patience_counter = 0
                print("patience counter:", patience_counter)
            elif curr_val_acc > min_val_acc:
                patience_counter += 1
                print("patience counter:", patience_counter)

            if patience_counter >= patience:
                break

        print( "curr_train_loss", curr_train_loss,
               "curr_train_acc", curr_train_acc,
               "curr_val_acc", curr_val_acc,
               "curr_test_acc", curr_test_acc)

        print("epoch ", epoch, "completed.")
    return epoch_train_losses,  epoch_val_acc, epoch_train_acc, epoch_test_acc


# Assumes cleaned up, padded sequences.
def eval(model, test_dataset, batch_size, device = "gpu"):
    '''

    Parameters
    ----------
    model
    test_dataset
    batch_size
    device

    Returns y_test, y_test_pred
    -------

    '''
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    scores = []
    device = torch.device("cuda:0" if torch.cuda.is_available() and device == "gpu" else "cpu")
    with torch.no_grad():
        for X_test, y_test, X_lens_test in test_loader:
            X_test = X_test.to(device)
            #X_lens_test = X_lens_test.to("cpu")
            ypred_test = model(X_test, X_lens_test)
            pred_scores = torch.sigmoid(ypred_test)
            scores.append(pred_scores)

    return torch.cat(scores), test_dataset.tensors[1]

def calc_metrics(pred_prob, y_test):
    '''
    This is done on CPU only.
    Parameters
    ----------
    pred_prob
    y_test

    Returns
    -------
    '''
    target_scores = y_test.cpu().numpy()
    pred_scores = pred_prob.detach().cpu().numpy() >= 0.5
    p_score = precision_score(target_scores, pred_scores)
    r_score = recall_score(target_scores, pred_scores)
    a_score = accuracy_score(target_scores, pred_scores)
    ra_score = roc_auc_score(target_scores, pred_scores)

    print("Precision:", p_score)
    print("Recall:", r_score)
    print("Accuracy:", a_score)
    print("ROC_AUC_Score:", ra_score)

    return p_score, r_score, a_score, ra_score

def plot_loss_acc(train_loss, val_loss, val_acc, output = "./output/" ):
    '''
    Visualize training loss vs. validation loss.
    Parameters
    ----------
    train_loss: training loss
    val_loss: validation loss

    Returns: None
    -------

    '''
    loss_csv = pd.DataFrame({"iter": range(len(train_loss)), "train_loss": train_loss,
                             "val_loss": val_loss, "val_acc": val_acc})
    loss_csv.to_csv(output + "loss.csv")
    # gca stands for 'get current axis'
    ax = plt.gca()
    loss_csv.plot(kind='line',x='iter',y='train_loss',ax=ax )
    loss_csv.plot(kind='line',x='iter',y='val_loss', color='red', ax=ax)
    # plt.show()
    plt.savefig(output + "train_vs_val_loss.png")

    plt.cla()
    #loss_csv.plot(kind='line', x='iter', y='train_acc', ax=ax)
    loss_csv.plot(kind='line', x='iter', y='val_acc', color='red', ax=ax)
    # plt.show()
    plt.savefig(output + "val_acc.png")
    plt.cla()
