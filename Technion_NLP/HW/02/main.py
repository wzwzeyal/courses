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
import argparse
import re

from nltk.corpus import stopwords

import nni
from nni.utils import merge_parameter

from DeepLearningFramework.Metric import Accuracy
from DeepLearningFramework.Training import TrainModel
from config import *


tqdm.pandas()

tokenizer = get_tokenizer("basic_english")

def strip_text(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    text = re.sub(r'https?:/\/\S+', ' ', text)
    return text.strip()

def review_clean_list(text):
    word_list = tokenizer(text)
    strip_words = [strip_text(word) for word in word_list]
    str_list = list(filter(None, strip_words))
    return str_list


class ClassificationDataset(Dataset):

    def __init__(self, file_path, glove_name, embeddimg_dim, stopwords, token_type):
        print(f'processing of {file_path} started ...')
        # Read data
        self.data_df = pd.read_csv(file_path).head(1024)
        global_vectors = GloVe(name=glove_name, dim=embeddimg_dim)

        pat = r'\b(?:{})\b'.format('|'.join(stopwords))
        # remove stop words
        print('removing stopwords ...')
        self.data_df['clean_review'] = self.data_df['review'].str.lower().replace(pat, '', regex=True)
        self.data_df['clean_review'] = self.data_df['clean_review'].str.replace(r'\s+', ' ', regex=True)

        self.token_type = token_type

        print('cleaning words ...')
        self.data_df['clean_review'] =  self.data_df.progress_apply(lambda x: review_clean_list(x['clean_review']), axis=1)

        print('calculating mean clean tokens ...')
        self.data_df['mean_clean_token'] =  self.data_df.progress_apply(lambda x: self.calc_mean_token(global_vectors, x['clean_review']), axis=1)

        print("tokenizing ...")
        tokenizer = get_tokenizer("basic_english")
        self.data_df['tokens'] = self.data_df.progress_apply(lambda x: tokenizer(x["review"]), axis=1)

        print("calculating mean tokens ...")

        self.data_df['mean_dirty_token'] = self.data_df.progress_apply(
            lambda x: self.calc_mean_token(global_vectors, x["tokens"]), axis=1
        )

        print('converting sentiment to number ...')
        self.data_df['target'] = pd.factorize(self.data_df['sentiment'], sort=True)[0]

        print(f'processing of {file_path} finished !')

    def __getitem__(self, item):
        return {
            'mean_token': self.data_df.iloc[item][self.token_type],
            'target': self.data_df.iloc[item]['target']}

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

    tokenizer = get_tokenizer("basic_english")
    stop = stopwords.words('english')
    train_dataset = ClassificationDataset(f'{DATA_PATH}/IMDB_train.csv', glove_name, embedding_dim, stop, token_type)
    test_dataset = ClassificationDataset(f'{DATA_PATH}/IMDB_test.csv', glove_name, embedding_dim, stop, token_type)

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
    parser.add_argument("--hidden_size_1", type=int, default=512, metavar='N',
                        help='hidden layer size (default: 512)')
    parser.add_argument("--hidden_size_2", type=int, default=512, metavar='N',
                        help='hidden layer size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='LR',
                        help='learning rate (default: 0.2)')
    parser.add_argument('--glove_args', type=str, default="840B|300", metavar='LR',
                        help='learning rate (default: "840B|300")')
    parser.add_argument('--token_type', type=str, default="mean_clean_token", metavar='LR',
                        help='token type (default: "840B|300")')

    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(get_params(), tuner_params))
    main(params)
