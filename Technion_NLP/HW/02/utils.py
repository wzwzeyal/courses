import re


def strip_text(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    text = re.sub(r'https?:/\/\S+', ' ', text)
    return text.strip()


def review_clean_list(tokenizer, text):
    word_list = tokenizer(text)
    strip_words = [strip_text(word) for word in word_list]
    str_list = list(filter(None, strip_words))
    return str_list
