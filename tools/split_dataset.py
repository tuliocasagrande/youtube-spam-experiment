import csv
import numpy as np
import os

DATA_FOLDER = 'data_split'

def read_dataset(filename):
    content_list = []
    label_list = []

    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        reader.next()  # Skipping the header

        for row in reader:
            content = row[3].replace('\n', ' ')
            content_list.append(content)

            label = int(row[4])
            label_list.append(label)

    X = np.asarray(content_list)
    y = np.asarray(label_list)

    return X, y


def read_unlabeled_dataset(filename):
    content_list = []

    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        reader.next()  # Skipping the header

        for row in reader:
            content = row[4].replace('\n', ' ')
            content_list.append(content)

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


def split_labeled_datasets(fileprefix):

    X, y = read_dataset(os.path.join('data_csv', fileprefix + '.csv'))
    X_pos_train, X_neg_train, X_pos_test, X_neg_test = split_dataset(X, y)

    with open(os.path.join(DATA_FOLDER, fileprefix + '-pos-train.txt'), 'w') as f:
        for each in X_pos_train:
            f.write('{0}\n'.format(each))

    with open(os.path.join(DATA_FOLDER, fileprefix + '-neg-train.txt'), 'w') as f:
        for each in X_neg_train:
            f.write('{0}\n'.format(each))

    with open(os.path.join(DATA_FOLDER, fileprefix + '-pos-test.txt'), 'w') as f:
        for each in X_pos_test:
            f.write('{0}\n'.format(each))

    with open(os.path.join(DATA_FOLDER, fileprefix + '-neg-test.txt'), 'w') as f:
        for each in X_neg_test:
            f.write('{0}\n'.format(each))


def get_corpus_file_list():
    return [corpus_file for corpus_file in os.listdir("youtube-corpus") if corpus_file.endswith(".csv")]


def join_unlabeled_corpus():
    corpus_file_list = get_corpus_file_list()

    with open(os.path.join(DATA_FOLDER, 'unsup.txt'), 'w') as f:
        for corpus_file in corpus_file_list:
            X_unsup = read_unlabeled_dataset(os.path.join('youtube-corpus', corpus_file))
            for each in X_unsup:
                f.write('{0}\n'.format(each))


if __name__ == '__main__':

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    file_list = ['01-9bZkp7q19f0',
                 '04-CevxZvSJLk8',
                 '07-KQ6zr6kCPj8',
                 '08-uelHwf8o7_U',
                 '09-pRpeEdMmmQ0']

    for fileprefix in file_list:
        split_labeled_datasets(fileprefix)

    join_unlabeled_corpus()
