import os
import gensim
import pickle
import numpy as np
import pandas as pd

def open_file(fname):
    with open(fname, "r") as f:
        data = f.read()

    return data


def save_pickle(fname, data):
    with open(fname, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    print(f"\npickle file was successfully saved: {fname}")


def load_pickle(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    print(f"\npickle file was successfully loaded: {fname}")

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


def load_pretrained_embedding(args):
    if args.pretrained_model == 'word2vec':
        pretrained_model_fname = os.path.join(args.data_path, "GoogleNews-vectors-negative300.bin.gz")
        pretrained_embedding_pickle_fname = os.path.join(args.data_path, "word2vec_embedding.pickle")

        if os.path.isfile(pretrained_embedding_pickle_fname):
            print("\nword2vec_embedding pickle was already exist")
            pretrained_model_index, pretrained_model_embedding = load_pickle(pretrained_embedding_pickle_fname)
        else:
            print("\nload pretrained word2vec model ... ")
            pretrained_model_index, pretrained_model_embedding = get_embedding_from_pretrained_model(pretrained_model_fname, pretrained_embedding_pickle_fname, args)

    elif args.pretrained_model == 'glove':
        pretrained_model_fname = os.path.join(args.data_path, "glove.6B.300d.txt")
        pretrained_embedding_pickle_fname = os.path.join(args.data_path, "glove_embedding.pickle")

        if os.path.isfile(pretrained_embedding_pickle_fname):
            print("\nglove_embedding pickle was already exist")
            pretrained_model_index, pretrained_model_embedding = load_pickle(pretrained_embedding_pickle_fname)
        else:
            print("\nload pretrained glove model ... ")
            pretrained_model_index, pretrained_model_embedding = get_embedding_from_pretrained_model(pretrained_model_fname, pretrained_embedding_pickle_fname, args)

    return pretrained_model_index, pretrained_model_embedding


def get_embedding_from_pretrained_model(pretrained_model_fname, pretrained_embedding_pickle_fname, args):
    if args.pretrained_model == 'word2vec':
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(pretrained_model_fname, binary=True)
        pretrained_model_index = {token: token_index for token_index, token in enumerate(word2vec.index2word)}
        pretrained_model_embedding = word2vec.vectors

    elif args.pretrained_model == 'glove':
        f = open(pretrained_model_fname)

        pretrained_model_index = {}
        pretrained_model_embedding = []

        for i, line in enumerate(f):
            line_split = line.split()
            word = line_split[0]
            pretrained_model_index[word] = i

            word_vector = line_split[1:]
            pretrained_model_embedding.append(word_vector)
        pretrained_model_embedding = np.array(pretrained_model_embedding, dtype=np.float)

    embedding_to_pickle = [pretrained_model_index, pretrained_model_embedding]
    save_pickle(pretrained_embedding_pickle_fname, embedding_to_pickle)

    print(f"\n{args.pretrained_model} embedding pickle was saved for future study")

    return pretrained_model_index, pretrained_model_embedding

