# This Python file uses the following encoding: utf-8

import numpy as np
import os
import random
import unicodecsv as csv

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

EXPERIMENT_FOLDER = 'exp-doc2vec'
if not os.path.exists(EXPERIMENT_FOLDER):
    os.makedirs(EXPERIMENT_FOLDER)

MODELS_FOLDER = 'doc2vec-models'
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)


EPOCH = 500
random.seed(1)  # for reproducibility


def get_corpus_file_list():
    return [corpus_file for corpus_file in os.listdir("youtube-corpus") if corpus_file.endswith(".csv")]


def read_dataset(filename):
    content_list = []
    label_list = []

    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        reader.next()  # Skipping the header

        for row in reader:
            content_list.append(row[3])
            label_list.append(int(row[4]))

    X = np.asarray(content_list)
    y = np.asarray(label_list)

    return X, y


def read_unlabeled_dataset(filename):
    content_list = []

    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        reader.next()  # Skipping the header

        for row in reader:
            content_list.append(row[4])

    X = np.asarray(content_list)

    return X


def split_dataset(X, y, train_percent=0.7):

    assert(len(X) == len(y))
    X_pos = X[y == 1]
    X_neg = X[y == 0]

    index_pos_train = int(len(X_pos) * train_percent)
    index_neg_train = int(len(X_neg) * train_percent)

    X_pos_train = X_pos[:index_pos_train]
    X_neg_train = X_neg[:index_neg_train]

    X_pos_test = X_pos[index_pos_train:]
    X_neg_test = X_neg[index_neg_train:]

    return X_pos_train, X_neg_train, X_pos_test, X_neg_test


def prepare_documents(sources):
    tokenizer = CountVectorizer().build_analyzer()
    for base, label in sources:
        for idx, sample in enumerate(base):
            yield TaggedDocument(tokenizer(sample), ['{}_{}'.format(label, idx)])


def doc2vec_vectorizer(sources, model_name, model):

    documents = [document for document in prepare_documents(sources)]

    model.build_vocab(documents)

    for epoch in range(EPOCH):
        logger.info('EPOCH: {}'.format(epoch))
        model.train(sorted(documents, key=lambda x: random.random()))

    model.save(os.path.join(MODELS_FOLDER, model_name))


if __name__ == "__main__":

    sources = ()
    X_unsup = []

    corpus_file_list = get_corpus_file_list()
    logger.info("YouTube corpus: {} videos".format(len(corpus_file_list)))

    for corpus_file in corpus_file_list:
        X_video_unsup = read_unlabeled_dataset(os.path.join('youtube-corpus', corpus_file))
        X_unsup = np.concatenate((X_unsup, X_video_unsup))

    sources += ((X_unsup, 'TRAIN_UNS'),)

    file_list = ['01-9bZkp7q19f0',
                 '04-CevxZvSJLk8',
                 '07-KQ6zr6kCPj8',
                 '08-uelHwf8o7_U',
                 '09-pRpeEdMmmQ0']

    for video_title in file_list:

        X, y = read_dataset(os.path.join('data_csv', video_title + '.csv'))
        X_pos_train, X_neg_train, X_pos_test, X_neg_test = split_dataset(X, y)

        sources += ((X_pos_train, 'TRAIN_POS_' + video_title),
                    (X_neg_train, 'TRAIN_NEG_' + video_title),
                    (X_pos_test, 'TEST_POS_' + video_title),
                    (X_neg_test, 'TEST_NEG_' + video_title),)

    logger.info("Documents ready! Available sources: {}".format(len(sources)))

    with open('doc2vec-labels.txt', 'w') as f:
        for base, label in sources:
            f.write("{}: {}\n".format(label, len(base)))

    models = [
        ('pv-dbow-s100.d2v',
         Doc2Vec(dm=0, window=5, size=100, negative=5, hs=0, min_count=1, workers=1, iter=1, seed=1)),
        ('pv-dbow-s300.d2v',
         Doc2Vec(dm=0, window=5, size=300, negative=5, hs=0, min_count=1, workers=1, iter=1, seed=1)),
    ]

    for model_name, model in models:
        doc2vec_vectorizer(sources, model_name, model)
