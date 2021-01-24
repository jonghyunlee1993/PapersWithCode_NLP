import re
import os
import codecs
import numpy as np
import pandas as pd

class Preprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.pos_data_path = os.path.join(self.data_path, 'rt-polarity.pos')
        self.neg_data_path = os.path.join(self.data_path, 'rt-polarity.neg')
        self.preced_data_path = os.path.join(self.data_path, 'polarity_df.csv')

    def build_data(self):
        reviews = []
        pos_data = codecs.open(self.pos_data_path, "r", encoding='utf-8', errors='ignore').read()
        neg_data = codecs.open(self.neg_data_path, "r", encoding='utf-8', errors='ignore').read()

        positive_reviews = [Preprocessor.clean_str(sent) for sent in pos_data.split('\n')[:-1]]
        negative_reviews = [Preprocessor.clean_str(sent) for sent in neg_data.split('\n')[:-1]]

        print(f'length of positive data: {len(positive_reviews)}\nlength of negative data: {len(negative_reviews)}\n'
              f'total length of data: {len(positive_reviews) + len(negative_reviews)}')

        for positive_review in positive_reviews:
            single_review = {'label': 1,
                     'text': positive_review,
                     'num_words': len(positive_review.split())
                     }
            reviews.append(single_review)

        for negative_review in negative_reviews:
            single_review = {'label': 0,
                     'text': negative_review,
                     'num_words': len(negative_review.split())
                     }
            reviews.append(single_review)

        word_to_idx = {'@pad': 0}

        for review in positive_reviews + negative_reviews:
            for word in review.split():
                if word not in word_to_idx:
                    word_to_idx[word] = len(word_to_idx)

        print(f'num of total data: {len(reviews)}\nnum of vocab: {len(word_to_idx)}')

        df = pd.DataFrame(reviews)
        df.to_csv(self.preced_data_path, index=False, encoding='utf-8')
        print(f'proced data was successfully saved: {self.preced_data_path}')

        return reviews, word_to_idx

    @staticmethod
    def clean_str(text):
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

