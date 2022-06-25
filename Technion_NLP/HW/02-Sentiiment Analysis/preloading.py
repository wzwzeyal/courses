import pandas as pd


def load_datasets(data_path):
    print("load_datasets Start")
    train_df = pd.read_csv(f'{data_path}/IMDB_train.csv')
    train_df['target'] = pd.factorize(train_df['sentiment'], sort=True)[0]
    print(f"loading {len(train_df)} train records")

    test_df = pd.read_csv(f'{data_path}/IMDB_test.csv')
    test_df['target'] = pd.factorize(test_df['sentiment'], sort=True)[0]
    print(f"loading {len(test_df)} test records")

    print("load_datasets Start")
    return train_df, test_df
