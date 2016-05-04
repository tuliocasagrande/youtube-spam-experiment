# This Python file uses the following encoding: utf-8

from classification import calculate_scores
import numpy as np
import os
import report
import unicodecsv as csv

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from nltk import word_tokenize as nltk_tokenizer
from sklearn.feature_extraction.text import CountVectorizer

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

EXPERIMENT_FOLDER = 'exp-doc2vec'
if not os.path.exists(EXPERIMENT_FOLDER):
    os.makedirs(EXPERIMENT_FOLDER)


MODELS_FOLDER = 'doc2vec_models'
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)


EPOCH = 5000


def read_dataset(video_title):
    content_list = []
    label_list = []

    # Reading and parsing CSV file
    filename = os.path.join('data_csv', video_title + '.csv')
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        reader.next()  # Skipping the header

        for row in reader:
            content_list.append(row[3])
            label_list.append(int(row[4]))

    X = np.asarray(content_list)
    y = np.asarray(label_list)

    return X, y


def split_dataset(X, y):

    assert(len(X) == len(y))
    X_pos = X[y == 1]
    X_neg = X[y == 0]

    train_percent = 0.7

    index_pos_train = int(len(X_pos) * train_percent)
    index_neg_train = int(len(X_neg) * train_percent)

    X_pos_train = X_pos[:index_pos_train]
    X_neg_train = X_neg[:index_neg_train]

    X_pos_test = X_pos[index_pos_train:]
    X_neg_test = X_neg[index_neg_train:]

    return X_pos_train, X_neg_train, X_pos_test, X_neg_test


def prepare_sentences(sources):
    tokenizer = CountVectorizer().build_analyzer()
    # tokenizer = nltk_tokenizer
    for base, label in sources:
        for idx, sample in enumerate(base):
            yield TaggedDocument(tokenizer(sample), ['{}_{}'.format(label, idx)])


def doc2vec_vectorizer(sources):

    sentences = [sentence for sentence in prepare_sentences(sources)]

    model = Doc2Vec(sentences, min_count=1, iter=EPOCH, workers=2)
    model.save(os.path.join(MODELS_FOLDER, 'corpus.d2v'))


if __name__ == "__main__":

    file_list = ['01-9bZkp7q19f0',
                 '04-CevxZvSJLk8',
                 '07-KQ6zr6kCPj8',
                 '08-uelHwf8o7_U',
                 '09-pRpeEdMmmQ0']

    sources = ()
    X_pos_train = []
    X_neg_train = []
    for video_title in file_list:

        X, y = read_dataset(video_title)
        X_pos_video_train, X_neg_video_train, X_pos_test, X_neg_test = split_dataset(X, y)

        X_pos_train = np.concatenate((X_pos_train, X_pos_video_train))
        X_neg_train = np.concatenate((X_neg_train, X_neg_video_train))

        sources += ((X_pos_test, 'TEST_POS_' + video_title),
                    (X_neg_test, 'TEST_NEG_' + video_title),)

    sources += ((X_pos_train, 'TRAIN_POS'),
                (X_neg_train, 'TRAIN_NEG'),)
    doc2vec_vectorizer(sources)
