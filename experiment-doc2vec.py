# This Python file uses the following encoding: utf-8

from classification import calculate_scores
import numpy as np
import os
import random
import report
import unicodecsv as csv

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

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
random.seed(1)  # for reproducibility


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


def prepare_documents(sources):
    tokenizer = CountVectorizer().build_analyzer()
    for base, label in sources:
        for idx, sample in enumerate(base):
            yield TaggedDocument(tokenizer(sample), ['{}_{}'.format(label, idx)])


def doc2vec_vectorizer(X, y):

    X_pos_train, X_neg_train, X_pos_test, X_neg_test = split_dataset(X, y)

    sources = (
        (X_pos_train, 'TRAIN_POS'),
        (X_neg_train, 'TRAIN_NEG'),
        (X_pos_test, 'TEST_POS'),
        (X_neg_test, 'TEST_NEG'),
    )

    documents = [document for document in prepare_documents(sources)]

    model = Doc2Vec(size=100, window=5, min_count=1, iter=1, seed=1, workers=1)
    model.build_vocab(documents)

    for epoch in range(EPOCH):
        logger.info('EPOCH: {}'.format(epoch))
        model.train(sorted(documents, key=lambda x: random.random()))

    model.save(os.path.join(MODELS_FOLDER, video_title + '.d2v'))

    vector_pos_train = [model.docvecs['TRAIN_POS_' + str(i)] for i in xrange(len(X_pos_train))]
    vector_neg_train = [model.docvecs['TRAIN_NEG_' + str(i)] for i in xrange(len(X_neg_train))]
    X_train = np.concatenate([vector_pos_train, vector_neg_train])

    label_pos_train = [1 for i in xrange(len(X_pos_train))]
    label_neg_train = [0 for i in xrange(len(X_neg_train))]
    y_train = np.concatenate([label_pos_train, label_neg_train])

    vector_pos_test = [model.docvecs['TEST_POS_' + str(i)] for i in xrange(len(X_pos_test))]
    vector_neg_test = [model.docvecs['TEST_NEG_' + str(i)] for i in xrange(len(X_neg_test))]
    X_test = np.concatenate([vector_pos_test, vector_neg_test])

    label_pos_test = [1 for i in xrange(len(X_pos_test))]
    label_neg_test = [0 for i in xrange(len(X_neg_test))]
    y_test = np.concatenate([label_pos_test, label_neg_test])

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    return X_train, y_train, X_test, y_test


def fit_and_predict(classifier, X_train, y_train, X_test):

    try:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
    except TypeError:
        logger.debug("TypeError:")
        logger.debug(TypeError)
        classifier.fit(X_train.toarray(), y_train)
        y_pred = classifier.predict(X_test.toarray())

    return y_pred


def get_best_params(classifier_title, classifier):
    if type(classifier) == GridSearchCV:
        best_parameters = classifier.best_estimator_.get_params()
        return classifier_title + ' - ' + ', '.join(
            ['{}: {}'.format(key, best_parameters[key])
             for key in classifier.param_grid]) + '\n'


def run_classifiers(X_train, y_train, X_test, y_test):

    # Parameters for grid search
    range5 = [10.0 ** i for i in range(-5, 5)]
    param_alpha = {'alpha': range5}
    param_C = {'C': range5}
    param_C_gamma = {'C': range5, 'gamma': range5}
    param_criterion = {'criterion': ['gini', 'entropy']}
    param_crit_nestim = {'criterion': ['gini', 'entropy'],
                         'n_estimators': range(10, 101, 10)}
    mcc = make_scorer(matthews_corrcoef)

    scores_list = []
    best_params = ''

    config = [
        # ('MultinomialNB',
        #  GridSearchCV(MultinomialNB(), param_alpha, cv=10, scoring=mcc)),
        ('BernoulliNB',
         GridSearchCV(BernoulliNB(), param_alpha, cv=10, scoring=mcc)),
        ('GaussianNB',
         GaussianNB()),
        ('SVM Linear',
         GridSearchCV(LinearSVC(), param_C, cv=10, scoring=mcc)),
        ('SVM RBF',
         GridSearchCV(SVC(kernel='rbf'), param_C_gamma, cv=10, scoring=mcc)),
        ('SVM Poly',
         GridSearchCV(SVC(kernel='poly'), param_C_gamma, cv=10, scoring=mcc)),
        ('Logistic',
         GridSearchCV(LogisticRegression(), param_C, cv=10, scoring=mcc)),
        ('DecisionTree',
         GridSearchCV(DecisionTreeClassifier(random_state=0), param_criterion, cv=10, scoring=mcc)),
        ('RandomForest',
         GridSearchCV(RandomForestClassifier(random_state=0), param_crit_nestim, cv=10, scoring=mcc)),
        ('1-NN',
         KNeighborsClassifier(n_neighbors=1)),
        ('3-NN',
         KNeighborsClassifier(n_neighbors=3)),
        ('5-NN',
         KNeighborsClassifier(n_neighbors=5))
    ]

    for classifier_title, classifier in config:
        logger.info("Fitting " + classifier_title)

        y_pred = fit_and_predict(classifier, X_train, y_train, X_test)
        scores_list.append((classifier_title, calculate_scores(y_test, y_pred)))
        best_params += get_best_params(classifier_title, classifier) or ''

    scores_list.sort(key=lambda scores: (scores[1]['mcc'], scores[1]['f1']),
                     reverse=True)
    return scores_list, best_params


if __name__ == "__main__":
    results_path = os.path.join(EXPERIMENT_FOLDER, 'results')
    figures_path = os.path.join(EXPERIMENT_FOLDER, 'figures')

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    file_list = ['01-9bZkp7q19f0',
                 '04-CevxZvSJLk8',
                 '07-KQ6zr6kCPj8',
                 '08-uelHwf8o7_U',
                 '09-pRpeEdMmmQ0']

    csv_filename = os.path.join(results_path, 'results_mcc.csv')
    clf_list = [
        # 'MultinomialNB',
        'BernoulliNB', 'GaussianNB', 'SVM Linear',
        'SVM RBF', 'SVM Poly', 'Logistic', 'DecisionTree',
        'RandomForest', '1-NN', '3-NN', '5-NN']
    csv_report = report.CsvReport(csv_filename, clf_list, 'mcc')

    with open(os.path.join(EXPERIMENT_FOLDER, 'best_params.txt'), 'w') as f:
        f.write('Best Parameters\n')

    for video_title in file_list:
        logger.info("TRAINING VIDEO " + video_title)

        X, y = read_dataset(video_title)
        X_train, y_train, X_test, y_test = doc2vec_vectorizer(X, y)
        scores_list, best_params = run_classifiers(X_train, y_train, X_test, y_test)

        with open(os.path.join(EXPERIMENT_FOLDER, 'best_params.txt'), 'a') as f:
            f.write('\n##############\n')
            f.write(video_title + '\n\n')
            f.write(best_params)

        tex_filename = os.path.join(results_path, video_title + '.tex')
        figurename = os.path.join(figures_path, video_title)

        report.tex_report(tex_filename, video_title, scores_list)
        report.plot_bars(figurename, video_title, scores_list, 'mcc')
        report.plot_roc(figurename, video_title, scores_list)
        csv_report.report(video_title, scores_list)
