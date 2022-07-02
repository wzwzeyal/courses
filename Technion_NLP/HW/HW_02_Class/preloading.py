import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import tqdm


def extract_names(base_url, suffix):
    names = []
    url = base_url.format(str(hex(suffix))[2:].upper())
    req = requests.get(url)
    soup = bs(req.text, 'html.parser')
    for t in soup.select('a[title*=פירוש]'):
        names.append(t.text)
    return names


def create_df(boys_names, girls_names):
    df1 = pd.DataFrame()
    df1['name'] = boys_names
    df1['gender'] = "boy"

    df2 = pd.DataFrame()
    df2['name'] = girls_names
    df2['gender'] = "girl"

    df = pd.concat([df1, df2])

    df3 = df[df.duplicated(["name"], keep=False)]
    df3['gender'] = 'UNISEX'
    df3.drop_duplicates(inplace=True)

    df = pd.concat([df, df3])

    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def load_datasets():
    boys_base_url = 'https://www.baby-names.co.il/category/%D7%9B%D7%9C-%D7%94%D7%A9%D7%9E%D7%95%D7%AA/%D7%A9%D7%9E%D7%95%D7%AA-%D7%9C%D7%91%D7%A0%D7%99%D7%9D/?ap=%D7%{}'
    girls_base_url = "https://www.baby-names.co.il/category/%D7%9B%D7%9C-%D7%94%D7%A9%D7%9E%D7%95%D7%AA/%D7%A9%D7%9E%D7%95%D7%AA-%D7%9C%D7%91%D7%A0%D7%95%D7%AA/?ap=%D7%{}"

    boys_names = []
    girls_names = []

    # for suffix in range(0x90, 0xAB):

    for suffix in range(0x90, 0xAB):
        boys_names.extend(extract_names(boys_base_url, suffix))
        print(len(boys_names))
        girls_names.extend(extract_names(girls_base_url, suffix))
        print(len(girls_names))

    df = create_df(boys_names, girls_names)
    return df
