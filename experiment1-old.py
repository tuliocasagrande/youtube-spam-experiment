# This Python file uses the following encoding: utf-8

from classification import SingleClassification
import os
import report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
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
    mcc = make_scorer(matthews_corrcoef)

    scores_list = []
    best_params = ''

    config = [
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
        ('1-NN',
         KNeighborsClassifier(n_neighbors=1)),
        ('3-NN',
         KNeighborsClassifier(n_neighbors=3)),
        ('5-NN',
         KNeighborsClassifier(n_neighbors=5))
    ]

    single_classification = SingleClassification(src_folder, video_title, TfidfVectorizer(use_idf=False))
    for classifier_title, classifier in config:
        logger.info("Fitting " + classifier_title)

        y_true, y_pred = single_classification.classify(classifier)
        scores_list.append((classifier_title, report.calculate_scores(y_true, y_pred)))
        best_params += get_best_params(classifier_title, classifier) or ''

    scores_list.sort(key=lambda scores: (scores[1]['mcc'], scores[1]['f1']),
                     reverse=True)
    return scores_list, best_params


def get_best_params(clf_title, classifier):
    if type(classifier) == GridSearchCV:
        best_parameters = classifier.best_estimator_.get_params()
        return clf_title + ' - ' + ', '.join(
            ['{}: {}'.format(key, best_parameters[key])
             for key in classifier.param_grid]) + '\n'


if __name__ == "__main__":
    EXPERIMENT_FOLDER = 'exp1_tf_normalized'
    SRC_FOLDER = 'data_split_normalized'

    results_path = os.path.join(EXPERIMENT_FOLDER, 'results')
    figures_path = os.path.join(EXPERIMENT_FOLDER, 'figures')

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    file_list = [('01-9bZkp7q19f0', 'Psy'),
                 ('04-CevxZvSJLk8', 'KatyPerry'),
                 ('07-KQ6zr6kCPj8', 'LMFAO'),
                 ('08-uelHwf8o7_U', 'Eminem'),
                 ('09-pRpeEdMmmQ0', 'Shakira')]

    csv_filename = os.path.join(results_path, 'results_mcc.csv')
    clf_list = ['MultinomialNB', 'BernoulliNB', 'GaussianNB', 'SVM Linear',
                'SVM RBF', 'SVM Poly', 'Logistic', 'DecisionTree',
                'RandomForest', '1-NN', '3-NN', '5-NN']
    csv_report = report.CsvReport(csv_filename, clf_list, 'mcc')

    with open(os.path.join(EXPERIMENT_FOLDER, 'best_params.txt'), 'w') as f:
        f.write('Best Parameters\n')

    for video_title, author in file_list:
        logger.info("TRAINING VIDEO " + video_title)
        scores_list, best_params = run_experiment(SRC_FOLDER, video_title)

        with open(os.path.join(EXPERIMENT_FOLDER, 'best_params.txt'), 'a') as f:
            f.write('\n##############\n')
            f.write(video_title + '\n\n')
            f.write(best_params)

        tex_filename = os.path.join(results_path, video_title + '.tex')
        figurename = os.path.join(figures_path, video_title)

        report.tex_report(tex_filename, video_title, scores_list, author)
        report.plot_bars(figurename, video_title, scores_list, 'mcc')
        report.plot_roc(figurename, video_title, scores_list)
        csv_report.report(video_title, scores_list)
