import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from tqdm import tqdm

import nni
import ast

from DeepLearningFramework.Metric import Accuracy
from DeepLearningFramework.Training import TrainModel
from config import *

tqdm.pandas()

import os


class ClassificationDataset(Dataset):

    def __init__(self, file_path, glove_name, embeddimg_dim):
        print(f'processing of {file_path} started ...')
        # Read data
        self.data_df = pd.read_csv(file_path).head(1024)

        print("tokenizing ...")
        tokenizer = get_tokenizer("basic_english")
        self.data_df['tokens'] = self.data_df.progress_apply(lambda x: tokenizer(x["review"]), axis=1)

        print("calculating mean tokens ...")
        global_vectors = GloVe(name=glove_name, dim=embeddimg_dim)
        self.data_df['mean_token'] = self.data_df.progress_apply(
            lambda x: self.calc_mean_token(global_vectors, x["tokens"]), axis=1
        )

        print('converting sentiment to number ...')
        self.data_df['target'] = pd.factorize(self.data_df['sentiment'], sort=True)[0]

        print(f'processing of {file_path} finished !')

    def __getitem__(self, item):
        return {'mean_token': self.data_df.iloc[item]['mean_token'], 'target': self.data_df.iloc[item]['target']}

    def __len__(self):
        return len(self.data_df)

    @staticmethod
    def calc_mean_token(global_vectors, wordlist):
        vectors = global_vectors.get_vecs_by_tokens(wordlist)
        return vectors.mean(dim=0)

    def get_targets(self):
        return self.data_df['target'].unique()


class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_dim, fc1_dim, fc2_dim, dropout, nof_classes):
        super(EmbeddingClassifier, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(embedding_dim, fc1_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(fc2_dim, nof_classes),

            # nn.Linear(fc1_dim, fc2_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            #
            # nn.Linear(fc2_dim, fc3_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            #
            # nn.Linear(fc3_dim, fc4_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            #
            # nn.Linear(fc4_dim, nof_classes),
        )

    def forward(self, x_batch):
        return self.seq(x_batch)


def epoch_callback(model, data_dl, criteria, metric, optimizer=None, scheduler=None, bTrain=True):
    epoch_loss = 0
    epoch_metric = 0
    count = 0
    n_iter = len(data_dl)
    lr_vector = np.full(n_iter, np.nan)
    DEVICE = next(model.parameters()).device  # -- CPU\GPU

    model.train(bTrain)  # -- train or test

    # -- Iterate over the mini-batches:
    for ii, loader_item_dict in enumerate(data_dl):
        # -- Move to device (CPU\GPU):
        X = loader_item_dict['mean_token'].to(DEVICE)
        Y = loader_item_dict['target'].to(DEVICE)

        # -- Forward:
        if bTrain:

            # -- Store computational graph:
            predictions = model(X)

            loss = criteria(predictions, Y)

        else:
            with torch.no_grad():
                # -- Do not store computational graph:
                predictions = model(X)
                loss = criteria(predictions, Y)

        # -- Backward:
        if bTrain:

            optimizer.zero_grad()  # -- set gradients to zeros
            loss.backward()  # -- backward
            optimizer.step()  # -- update parameters
            if scheduler is not None:
                lr_vector[ii] = scheduler.get_last_lr()[0]
                scheduler.step()  # -- update learning rate

        with torch.no_grad():

            Nb = X.shape[0]
            count += Nb
            epoch_loss += Nb * loss.item()
            epoch_metric += Nb * metric(predictions, Y)
        print(f'\r{"Train" if bTrain else "Val"} - Iteration: {ii:3d} ({n_iter}): loss = {loss:2.6f}', end='')

    print('', end='\r')
    epoch_loss /= count
    epoch_metric /= count

    # nni.report_intermediate_result(epoch_loss)
    nni.report_intermediate_result(epoch_metric)

    return epoch_loss, epoch_metric, lr_vector


def main(args):
    print("HW2 Start")
    print(os.getcwd())

    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    print(args)

    # configure hyper-parameters
    glove_args = args["glove_args"].split("|")
    glove_name = glove_args[0]
    embedding_dim = int(glove_args[1])
    batch_size = args["batch_size"]
    hidden_size_1 = args["hidden_size_1"]
    hidden_size_2 = args["hidden_size_2"]
    dropout = args["dropout"]
    lr = args["lr"]

    train_dataset = ClassificationDataset(f'{DATA_PATH}/IMDB_train.csv', glove_name, embedding_dim)
    test_dataset = ClassificationDataset(f'{DATA_PATH}/IMDB_test.csv', glove_name, embedding_dim)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    target_classes = train_dataset.get_targets()

    n_epochs = NOF_EPOCHS
    n_iter = n_epochs * len(train_loader)

    loss = nn.CrossEntropyLoss()
    metric = Accuracy()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    o_model = EmbeddingClassifier(
        embedding_dim,
        fc1_dim=hidden_size_1,
        fc2_dim=hidden_size_2,
        dropout=dropout,
        nof_classes=len(target_classes)).to(DEVICE)

    o_optim = optim.AdamW(o_model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-2)
    o_scheduler = OneCycleLR(o_optim, max_lr=lr, total_steps=n_iter)

    trial_hostory = TrainModel(o_model, train_loader, test_loader, loss, metric, n_epochs, o_optim, o_scheduler, Epoch=epoch_callback,
               sModelName='EmbeddingCLS')

    vTrainLoss, vTrainAcc, vValLoss, vValAcc, vLR = trial_hostory
    nni.report_final_result(vValAcc  .max())

    # print(lHistory)

    print("HW2 End")


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    tuner_params = nni.get_next_parameter()
    main(tuner_params)
