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


def main():
    print(ord(u"א"))

    # URL = 'https://www.baby-names.co.il/category/%D7%9B%D7%9C-%D7%94%D7%A9%D7%9E%D7%95%D7%AA/%D7%A9%D7%9E%D7%95%D7%AA-%D7%9C%D7%91%D7%A0%D7%99%D7%9D/?ap=%D7%90'

    df = load_datasets()

    df.to_csv('dataset.csv')
    print(len(df))

    #
    #
    #
    # for n in range(0x90, 0xAB):
    #     c = str(hex(n))[2:].upper()
    #     print(c)

    # for a in soup.find_all('a', href=True):
    #     print(a)

    # print(titles[4].text)







if __name__ == '__main__':
    # execute only if run as the entry point into the program

    # When executed using the NNI (https://github.com/microsoft/nni) framework
    # it get the next set of hyper parameters
    # nni create --config config_windows.yml
    # tuner_params = nni.get_next_parameter()
    # params = vars(merge_parameter(get_params(), tuner_params))
    # main(params)
    main()


