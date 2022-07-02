import argparse

import nltk
import nni
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nni.utils import merge_parameter
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from tqdm import tqdm

from DeepLearningFramework.Auxiliary import PlotHistory
from DeepLearningFramework.Metric import Accuracy
from DeepLearningFramework.Training import TrainModel
from config import *
from utils import review_clean_list

tqdm.pandas()


class ClassificationDataset(Dataset):
    """

    """
    def __init__(self, file_path, glove_name, embedding_dim, stop_words, token_type):
        """

        :param file_path: path to the csv data file
        :param glove_name: the name of the global vectors to use (e.g. 840B)
        :param embedding_dim: the embedding dimension (e.g. 300)
        :param stop_words:  a list of stop words
        :param token_type: use clean or raw (mean_clean_token, mean_dirty_token)
        """
        print(f'processing of {file_path} started ...')

        tokenizer = get_tokenizer("basic_english")
        global_vectors = GloVe(name=glove_name, dim=embedding_dim)

        # Read data
        self.data_df = pd.read_csv(file_path)

        # remove stop words
        print('removing stopwords ...')
        pat = r'\b(?:{})\b'.format('|'.join(stop_words))
        self.data_df['clean_review'] = self.data_df['review'].str.lower().replace(pat, '', regex=True)
        self.data_df['clean_review'] = self.data_df['clean_review'].str.replace(r'\s+', ' ', regex=True)

        print('cleaning words ...')
        self.data_df['clean_review'] = self.data_df.progress_apply(
            lambda x: review_clean_list(tokenizer, x['clean_review']), axis=1)

        print('calculating mean clean tokens ...')

        self.data_df['mean_clean_token'] = self.data_df.progress_apply(
            lambda x: self.calc_mean_token(global_vectors, x['clean_review']), axis=1)

        print("tokenizing ...")
        self.data_df['tokens'] = self.data_df.progress_apply(lambda x: tokenizer(x["review"]), axis=1)

        print("calculating mean tokens ...")
        self.data_df['mean_dirty_token'] = self.data_df.progress_apply(
            lambda x: self.calc_mean_token(global_vectors, x["tokens"]), axis=1
        )

        print('converting sentiment to number ...')
        self.data_df['target'] = pd.factorize(self.data_df['sentiment'], sort=True)[0]

        print(f'processing of {file_path} finished !')

        self.token_type = token_type

    def __getitem__(self, item):
        return {
            'mean_token': self.data_df.iloc[item][self.token_type],
            'target': self.data_df.iloc[item]['target']}

    def __len__(self):
        return len(self.data_df)

    @staticmethod
    def calc_mean_token(global_vectors, wordlist):
        """
        Tokenizes the word list using the global_vectors and
        returns its mean value (e.g. ["hello", "world"] -> [[2, 10, 4], [1, 8, 3]] -> [0.5, 9, 3.5]
        :param global_vectors:
        :param wordlist:
        :return: mean value in the shape of the embedding dimension
        """
        vectors = global_vectors.get_vecs_by_tokens(wordlist)
        return vectors.mean(dim=0)

    def get_targets(self):
        return self.data_df['target'].unique()


class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_dim, fc1_dim, fc2_dim, dropout, nof_classes):
        """

        :param embedding_dim: the embedding dimension (e.g. 300)
        :param fc1_dim: first hidden layer size
        :param fc2_dim: second hidden layer size
        :param dropout: dropout factor
        :param nof_classes: number of classes (e.g. 2)
        """
        super(EmbeddingClassifier, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(embedding_dim, fc1_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(fc2_dim, nof_classes),
        )

    def forward(self, x_batch):
        return self.seq(x_batch)


def epoch_callback(model, data_dl, criteria, metric, optimizer=None, scheduler=None, bTrain=True):
    """
    This function perform the model training loop
    :param model:
    :param data_dl:
    :param criteria:
    :param metric:
    :param optimizer:
    :param scheduler:
    :param bTrain:
    :return: model training statstics (epoch_loss, epoch_metric, lr_vector)
    """
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
    """
    HW 2 Overview code documentation

    This is the main function
    It receives the hyper parameters as arguments

    1. Creates the data sets
    2. Creates the data loaders
    3. Builds the classifying model
    4. Train the model
    5. Plots and save the results

    :param args: the hyper parameters for the model.
    :return:
    """
    print("HW2 Start")

    nltk.download('stopwords')

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
    token_type = args["token_type"]

    stop_words = stopwords.words('english')

    train_dataset = ClassificationDataset(f'{DATA_PATH}/IMDB_train.csv', glove_name, embedding_dim, stop_words, token_type)
    test_dataset = ClassificationDataset(f'{DATA_PATH}/IMDB_test.csv', glove_name, embedding_dim, stop_words, token_type)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    target_classes = train_dataset.get_targets()

    n_epochs = NOF_EPOCHS
    n_iter = n_epochs * len(train_loader)

    loss = nn.CrossEntropyLoss()
    metric = Accuracy()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # DEVICE = "cpu"

    print(DEVICE)

    o_model = EmbeddingClassifier(
        embedding_dim,
        fc1_dim=hidden_size_1,
        fc2_dim=hidden_size_2,
        dropout=dropout,
        nof_classes=len(target_classes)).to(DEVICE)

    o_optim = optim.AdamW(o_model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-2)
    o_scheduler = OneCycleLR(o_optim, max_lr=lr, total_steps=n_iter)

    trial_hostory = TrainModel(o_model, train_loader, test_loader, loss, metric, n_epochs, o_optim, o_scheduler,
                               Epoch=epoch_callback,
                               sModelName='EmbeddingCLS')

    vTrainLoss, vTrainAcc, vValLoss, vValAcc, vLR = trial_hostory
    nni.report_final_result(vValAcc.max())

    PlotHistory(trial_hostory)
    plt.savefig('hw2_results.png')

    print("HW2 End")


def get_params():
    # Training settings

    # {
    #     "batch_size": {"_type": "choice", "_value": [64, 128]},
    #     "hidden_size_1": {"_type": "choice", "_value": [256, 512, 1024]},
    #     "hidden_size_2": {"_type": "choice", "_value": [256, 64]},
    #     "lr": {"_type": "choice", "_value": [0.001, 0.01]},
    #     "dropout": {"_type": "choice", "_value": [0.0, 0.2]},
    #     "glove_args": {"_type": "choice", "_value": ["840B|300"]}
    # }
    parser = argparse.ArgumentParser(description='HW2')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--hidden_size_1", type=int, default=256, metavar='N',
                        help='hidden layer 1 size (default: 256)')
    parser.add_argument("--hidden_size_2", type=int, default=64, metavar='N',
                        help='hidden layer 2 size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='LR',
                        help='dropout  (default: 0.01)')
    parser.add_argument('--glove_args', type=str, default="840B|300", metavar='LR',
                        help='glove_args rate (default: "840B|300")')
    parser.add_argument('--token_type', type=str, default="mean_dirty_token", metavar='LR',
                        help='token type (default: "mean_dirty_token")')

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    # execute only if run as the entry point into the program

    # When executed using the NNI (https://github.com/microsoft/nni) framework
    # it get the next set of hyper parameters
    # nni create --config config_windows.yml
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(get_params(), tuner_params))
    main(params)
