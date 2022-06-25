from Technion_NLP.HW2.config import DATA_FILES
from Technion_NLP.HW2.preloading import load_datasets

# https://towardsdatascience.com/sentiment-analysis-on-a-imdb-movie-review-dataset-with-a-support-vector-machines-model-in-python-50c1d487327e


def main():
    print("HW2 Start")
    load_datasets(DATA_FILES)
    print("HW2 End")


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()