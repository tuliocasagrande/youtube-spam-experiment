import csv
import numpy as np
import os


# This is a chronologically sorted stratified holdout 70/30
# 70% for training and 30% for testing
def split(fileprefix):

    content_list = []
    label_list = []

    with open(os.path.join('data_csv', fileprefix+'.csv'), 'rb') as csvfile:
        reader = csv.reader(csvfile)
        reader.next()  # Skipping the header

        for row in reader:
            content_list.append(row[3])
            label_list.append(int(row[4]))

    X = np.asarray(content_list)
    y = np.asarray(label_list)

    assert(len(X) == len(y))

    X_spam = X[y == 1]
    y_spam = y[y == 1]
    X_ham = X[y == 0]
    y_ham = y[y == 0]

    index_spam = int(len(X_spam) * 0.7)
    index_ham = int(len(X_ham) * 0.7)

    X_train = np.concatenate([X_spam[:index_spam], X_ham[:index_ham]])
    y_train = np.concatenate([y_spam[:index_spam], y_ham[:index_ham]])

    X_test = np.concatenate([X_spam[index_spam:], X_ham[index_ham:]])
    y_test = np.concatenate([y_spam[index_spam:], y_ham[index_ham:]])

    assert(len(X_train) == len(y_train))
    assert(len(X_test) == len(y_test))

    with open(os.path.join('data_split', fileprefix+'_train'), 'w') as f:
        for i in xrange(len(X_train)):
            f.write('{0},{1}\n'.format(y_train[i], X_train[i]))

    with open(os.path.join('data_split', fileprefix+'_test'), 'w') as f:
        for each in X_test:
            f.write('{0}\n'.format(each))

    with open(os.path.join('data_split', fileprefix+'_goldstandard'), 'w') as f:
        for each in y_test:
            f.write('{0}\n'.format(each))


if __name__ == '__main__':

    if not os.path.exists('data_split'):
        os.makedirs('data_split')

    file_list = ['01-9bZkp7q19f0',
                 '04-CevxZvSJLk8',
                 '07-KQ6zr6kCPj8',
                 '08-uelHwf8o7_U',
                 '09-pRpeEdMmmQ0']

    for fileprefix in file_list:
        split(fileprefix)
