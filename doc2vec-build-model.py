# This Python file uses the following encoding: utf-8

import os
import random

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

MODELS_FOLDER = 'doc2vec-models'
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)


EPOCH = 100
random.seed(1)  # for reproducibility


def read_dataset(src_folder, video_title):

    pos_train = open(os.path.join(src_folder, video_title + '-pos-train.txt')).readlines()
    neg_train = open(os.path.join(src_folder, video_title + '-neg-train.txt')).readlines()
    pos_test = open(os.path.join(src_folder, video_title + '-pos-test.txt')).readlines()
    neg_test = open(os.path.join(src_folder, video_title + '-neg-test.txt')).readlines()

    return pos_train, neg_train, pos_test, neg_test


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

    SRC_FOLDER = 'data_split'
    sources = ()

    X_unsup = open(os.path.join(SRC_FOLDER, 'unsup.txt')).readlines()
    sources += ((X_unsup, 'TRAIN_UNS'),)

    file_list = ['01-9bZkp7q19f0',
                 '04-CevxZvSJLk8',
                 '07-KQ6zr6kCPj8',
                 '08-uelHwf8o7_U',
                 '09-pRpeEdMmmQ0']

    for video_title in file_list:

        pos_train, neg_train, pos_test, neg_test = read_dataset(SRC_FOLDER, video_title)

        sources += ((pos_train, 'TRAIN_POS_' + video_title),
                    (neg_train, 'TRAIN_NEG_' + video_title),
                    (pos_test, 'TEST_POS_' + video_title),
                    (neg_test, 'TEST_NEG_' + video_title),)

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
