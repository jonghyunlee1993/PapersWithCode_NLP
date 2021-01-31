import gensim
import pickle
import pandas as pd


def open_file(fname):
    with open(fname, "r") as f:
        data = f.read()

    return data


def save_pickle(fname, data):
    with open(fname, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    print(f"pickle file was successfully saved: {fname}")


def load_pickle(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    print(f"pickle file was successfully loaded: {fname}")

    return data


def save_csv(fname, data):
    print(data)
    df = pd.DataFrame(data)
    df.to_csv(fname, index=False, encoding='utf-8')


def generate_review_data(start_index, label, sentences, reviews):
    for i, sent in enumerate(sentences):
        datum = {'label': label,
                 'text': sent}
        reviews[start_index + i] = datum

    return reviews


def load_pretrained_word2vec(word2vec_path):
    word2vec = gensim.models.keyedvectors._load_word2vec_format(word2vec_path, binary=True)
    word2vec_index = get_word2vec_index(word2vec)

    return word2vec_index


def get_word2vec_index(word2vec):
    return {token: token_index for token_index, token in enumerate(word2vec.index2word)}