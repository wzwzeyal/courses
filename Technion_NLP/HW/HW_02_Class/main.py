# Boys
# https://www.baby-names.co.il/category/%D7%9B%D7%9C-%D7%94%D7%A9%D7%9E%D7%95%D7%AA/%D7%A9%D7%9E%D7%95%D7%AA-%D7%9C%D7%91%D7%A0%D7%99%D7%9D/?ap=%D7%90
# https://www.baby-names.co.il/category/%D7%9B%D7%9C-%D7%94%D7%A9%D7%9E%D7%95%D7%AA/%D7%A9%D7%9E%D7%95%D7%AA-%D7%9C%D7%91%D7%A0%D7%99%D7%9D/?ap=%D7%91
# .
# .
# .
# https://www.baby-names.co.il/category/%D7%9B%D7%9C-%D7%94%D7%A9%D7%9E%D7%95%D7%AA/%D7%A9%D7%9E%D7%95%D7%AA-%D7%9C%D7%91%D7%A0%D7%99%D7%9D/?ap=%D7%AA

# Girls
# https://www.baby-names.co.il/category/%D7%9B%D7%9C-%D7%94%D7%A9%D7%9E%D7%95%D7%AA/%D7%A9%D7%9E%D7%95%D7%AA-%D7%9C%D7%91%D7%A0%D7%95%D7%AA/?ap=%D7%90
# .
# .
# .
# https://www.baby-names.co.il/category/%D7%9B%D7%9C-%D7%94%D7%A9%D7%9E%D7%95%D7%AA/%D7%A9%D7%9E%D7%95%D7%AA-%D7%9C%D7%91%D7%A0%D7%95%D7%AA/?ap=%D7%AA

import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import tqdm

from preloading import load_datasets
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def main():
    print(ord(u"א"))

    # URL = 'https://www.baby-names.co.il/category/%D7%9B%D7%9C-%D7%94%D7%A9%D7%9E%D7%95%D7%AA/%D7%A9%D7%9E%D7%95%D7%AA-%D7%9C%D7%91%D7%A0%D7%99%D7%9D/?ap=%D7%90'

    df = load_datasets()

    df.to_csv('dataset.csv')

    n_iters = 100000

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        train()
    print(len(df))

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn


def train(category_tensor, line_tensor):
    criterion = nn.NLLLoss()
    rnn = RNN(10, 11, 12)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


if __name__ == '__main__':
    # execute only if run as the entry point into the program

    # When executed using the NNI (https://github.com/microsoft/nni) framework
    # it get the next set of hyper parameters
    # nni create --config config_windows.yml
    # tuner_params = nni.get_next_parameter()
    # params = vars(merge_parameter(get_params(), tuner_params))
    # main(params)
    main()
