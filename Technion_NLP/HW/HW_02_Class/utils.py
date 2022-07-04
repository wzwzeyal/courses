import torch
import unicodedata
import string

import time
import math

all_letters = "אבגדהוזחטיכךלמםנןסעפףצץקרשת" + " .,;'"

n_letters = len(all_letters)


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for i, letter in enumerate(line):
        tensor[i][0][letterToIndex(letter)] = 1
    return tensor


import random


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample(row):
    """
    return a random example from the dataset
    :param category_lines:
    :param all_categories: ["Boy", "Girl", "Unisex"]
    :return:
    """

    row_values = row.iloc[0]

    # Get Category Boy/Girl/Unisex
    category = row_values["gender"]
    name = row_values["name"]

    category_tensor = torch.tensor([row_values["target"]], dtype=torch.long)
    line_tensor = lineToTensor(name)
    return category, name, category_tensor, line_tensor


def categoryFromOutput(all_categories, output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)