import os
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


def load_pretrained_word2vec(data_path):
    pretrained_model_fname = os.path.join(data_path, "GoogleNews-vectors-negative300.bin.gz")
    word2vec_index_fname   = os.path.join(data_path, "word2vec_index.pickle")

    if os.path.isfile(word2vec_index_fname):
        print("word2vec_index pickle was already exist")
        word2vec_index = load_pickle(word2vec_index_fname)
    else:
        print("\nload pretrained word2vec model ... ")
        word2vec = gensim.models.keyedvectors.load_word2vec_format(pretrained_model_fname, binary=True)
        word2vec_index = get_word2vec_index(word2vec)
        save_pickle(word2vec_index_fname, word2vec_index)
        print("\nword2vec_index pickle was saved for future study")

    return word2vec_index


def get_word2vec_index(word2vec):
    return {token: token_index for token_index, token in enumerate(word2vec.index2word)}