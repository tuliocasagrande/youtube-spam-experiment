# This Python file uses the following encoding: utf-8

from classification import SingleClassification
import os
import report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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


def run_experiment(src_folder, video_title):

    # Parameters for grid search
    range5 = [10.0 ** i for i in range(-5, 5)]
    param_alpha = {'alpha': range5}
    param_C = {'C': range5}
    param_C_gamma = {'C': range5, 'gamma': range5}
    param_criterion = {'criterion': ['gini', 'entropy']}
    param_crit_nestim = {'criterion': ['gini', 'entropy'],
                         'n_estimators': range(10, 101, 10)}
    param_neighbors = {'n_neighbors': [1, 3, 5, 7, 9]}
    mcc = make_scorer(matthews_corrcoef)

    scores_list = []
    best_params = ''

    CONFIG = [
        ('MultinomialNB',
         GridSearchCV(MultinomialNB(), param_alpha, cv=10, scoring=mcc)),
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
        ('k-NN',
         GridSearchCV(KNeighborsClassifier(), param_neighbors, cv=10, scoring=mcc)),
    ]

    count = CountVectorizer()
    tfidf = TfidfVectorizer()
    tf = TfidfVectorizer(use_idf=False)
    single_classification = SingleClassification(src_folder, video_title, count)
    for classifier_title, classifier in CONFIG:
        logger.info("Fitting " + classifier_title)

        y_true, y_pred = single_classification.classify(classifier)
        scores_list.append((classifier_title, report.calculate_scores(y_true, y_pred)))
        best_params += get_best_params(classifier_title, classifier) or ''

    scores_list.sort(key=lambda scores: (scores[1]['mcc'], scores[1]['f1']),
                     reverse=True)

    with open(os.path.join(RESULTS_PATH, 'best_params.txt'), 'a') as f:
        f.write('\n##############\n' + video_title + '\n\n' + best_params)

    report.csv_report(RESULTS_PATH, video_title, scores_list)


def get_best_params(clf_title, classifier):
    if type(classifier) == GridSearchCV:
        best_parameters = classifier.best_estimator_.get_params()
        return clf_title + ' - ' + ', '.join(
            ['{}: {}'.format(key, best_parameters[key])
             for key in classifier.param_grid]) + '\n'


if __name__ == "__main__":
    # RESULTS_SUBFOLDER = 'tf_normalized'
    # SRC_FOLDER = 'data_split_normalized'
    RESULTS_SUBFOLDER = 'exp1_count'
    SRC_FOLDER = 'data_split'

    # RESULTS_PATH = os.path.join('results', RESULTS_SUBFOLDER)
    RESULTS_PATH = os.path.join(RESULTS_SUBFOLDER, 'results')
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    file_list = ['01-9bZkp7q19f0',
                 '04-CevxZvSJLk8',
                 '07-KQ6zr6kCPj8',
                 '08-uelHwf8o7_U',
                 '09-pRpeEdMmmQ0']

    with open(os.path.join(RESULTS_PATH, 'best_params.txt'), 'w') as f:
        f.write('Best Parameters\n')

    for video_title in file_list:
        logger.info("TRAINING VIDEO " + video_title)
        run_experiment(SRC_FOLDER, video_title)
