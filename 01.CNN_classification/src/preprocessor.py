import re
import os
import argparse

import util

class Preprocessor:
    def __init__(self, args):
        self.args = args
        self.data_path = self.args.data_path
        self.pos_data = os.path.join(self.data_path, self.args.pos_data)
        self.neg_data = os.path.join(self.data_path, self.args.neg_data)
        self.result = os.path.join(self.data_path, self.args.preprocessed_data)

    def build_data(self):
        if not  os.path.isfile(self.result):
            pos = util.open_file(self.pos_data)
            neg = util.open_file(self.neg_data)

            pos_sentences = [self.clean_str(sent) for sent in pos.split('\n')[:-1]]
            neg_sentences = [self.clean_str(sent) for sent in neg.split('\n')[:-1]]

            pos_length = len(pos_sentences)
            neg_length = len(neg_sentences)
            tot_length = pos_length + neg_length

            print(f'Positive: {pos_length}\nNegative: {neg_length}\nTotal: {tot_length}')

            reviews = [0] * tot_length

            reviews = util.generate_review_data(1, pos_sentences, reviews)
            reviews = util.generate_review_data(0, neg_sentences, reviews)

            word_to_idx = {'@pad': 0}

            for sentence in pos_sentences + neg_sentences:
                for word in sentence.split():
                    if word not in word_to_idx:
                        word_to_idx[word] = len(word_to_idx)

            data_to_pickle = [reviews, word_to_idx]

            util.save_pickle(self.result, data_to_pickle)
        else:
            reviews, word_to_idx = util.load_pickle(self.result)

        print(f'number of reviews: {len(reviews)}\nnumber of vocabularies: {len(word_to_idx)}')

        return reviews, word_to_idx

    def clean_str(self, text):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Every dataset is lower cased except for TREC
        """
        text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
        text = re.sub(r"\'s", " \'s", text)
        text = re.sub(r"\'ve", " \'ve", text)
        text = re.sub(r"n\'t", " n\'t", text)
        text = re.sub(r"\'re", " \'re", text)
        text = re.sub(r"\'d", " \'d", text)
        text = re.sub(r"\'ll", " \'ll", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\(", " \( ", text)
        text = re.sub(r"\)", " \) ", text)
        text = re.sub(r"\?", " \? ", text)
        text = re.sub(r"\s{2,}", " ", text)

        return text.strip().lower()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='text preprocessing module')
    parser.add_argument('--data_path', type=str, default='../data', help='data folder')
    parser.add_argument('--pos_data', type=str, default='rt-polarity.pos', help='file name of positive raw data')
    parser.add_argument('--neg_data', type=str, default='rt-polarity.neg', help='file name of negative raw data')
    parser.add_argument('--preprocessed_data', type=str, default='polarity_data.pickle', help='file name of processed data')
    args = parser.parse_args()

    preprocessor = Preprocessor(args)
    preprocessor.build_data()
