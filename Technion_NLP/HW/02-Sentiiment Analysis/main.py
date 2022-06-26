from config import DATA_FILES
from preloading import load_datasets

# https://towardsdatascience.com/sentiment-analysis-on-a-imdb-movie-review-dataset-with-a-support-vector-machines-model-in-python-50c1d487327e
# https://github.com/FixelAlgorithmsTeam/FixelCourses/tree/9aaefce077927432f1cd62906dba8f25e5ef0090

def main():
    print("HW2 Start")
    train_df, test_df = load_datasets(DATA_FILES)
    print("HW2 End")


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()