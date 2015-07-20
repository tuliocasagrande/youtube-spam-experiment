#!/usr/bin/python
# This Python file uses the following encoding: utf-8

from classification import calculate_scores, SingleClassification, DualClassification, SemiSupervisedClassification
import os
import report
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


def exp2(filename):

  # Parameters for grid search
  range5 = [10.0 ** i for i in range(-5,5)]
  param_alpha = {'alpha': range5}
  param_gamma = {'gamma': range5}
  param_C = {'C': range5}
  param_C_gamma = {'C': range5, 'gamma': range5}
  param_criterion = {'criterion': ['gini', 'entropy']}
  param_crit_nestim = {'criterion': ['gini', 'entropy'], 'n_estimators': range(10,101,10)}
  mcc = make_scorer(matthews_corrcoef)

  scores_list = []

  svm_grid = GridSearchCV(LinearSVC(), param_C, cv=10, scoring=mcc)
  config = [('SVM 0.3', SingleClassification(filename, train_percent=0.3, test_percent=0.3)),
            ('SVM 0.4', SingleClassification(filename, train_percent=0.4, test_percent=0.3)),
            ('SVM 0.5', SingleClassification(filename, train_percent=0.5, test_percent=0.3)),
            ('SVM 0.6', SingleClassification(filename, train_percent=0.6, test_percent=0.3)),
            ('SVM 0.7', SingleClassification(filename, train_percent=0.7))]

  for clf_title, option in config:
    y_true, y_pred, clf = option.classify(svm_grid);
    scores_list.append((clf_title, calculate_scores(y_true, y_pred)))
    print clf_title
    print_best_params(clf)

  ss_clf = LabelSpreading(kernel='rbf', gamma=1)
  # ss_grid = GridSearchCV(LabelSpreading(kernel='rbf'), param_gamma, cv=10)
  config = [('SVM 0.3 + SS 0.4', SemiSupervisedClassification(filename, threshold=0.9, train_percent=0.3, ss_percent=0.4)),
            ('SVM 0.4 + SS 0.3', SemiSupervisedClassification(filename, threshold=0.9, train_percent=0.4, ss_percent=0.3)),
            ('SVM 0.5 + SS 0.2', SemiSupervisedClassification(filename, threshold=0.9, train_percent=0.5, ss_percent=0.2)),
            ('SVM 0.6 + SS 0.1', SemiSupervisedClassification(filename, threshold=0.9, train_percent=0.6, ss_percent=0.1))]

  for clf_title, option in config:
    y_true, y_pred, interm_clf, final_clf, len_above_SS, len_X_ss = option.classify(ss_clf, svm_grid);
    # scores_list.append(('{0} ({1}/{2})'.format(clf_title, len_above_SS, len_X_ss), calculate_scores(y_true, y_pred)))
    scores_list.append((clf_title, calculate_scores(y_true, y_pred)))
    print clf_title
    print_best_params(interm_clf)
    print_best_params(final_clf)

  scores_list.sort(key=lambda scores: (scores[1]['mcc'], scores[1]['f1']), reverse=True)
  return scores_list


def print_best_params(clf):
  if type(clf) != GridSearchCV:
    print clf.__class__.__name__
  else:
    print clf.best_estimator_.__class__.__name__
    best_parameters = clf.best_estimator_.get_params()
    for key in clf.param_grid:
      print '\t{0}: {1}'.format(key, best_parameters[key])


if __name__ == "__main__":
  results_path = os.path.join('exp2', 'results')
  figures_path = os.path.join('exp2', 'figures')

  if not os.path.exists(results_path):
    os.makedirs(results_path)
  if not os.path.exists(figures_path):
    os.makedirs(figures_path)

  file_list = ['01-PSY-9bZkp7q19f0',
               '04-KatyPerry-CevxZvSJLk8',
               '07-LMFAO-KQ6zr6kCPj8',
               # '08-Eminem-uelHwf8o7_U',
               '08-rotulada-tratada-embaralhada',
               '09-Shakira-pRpeEdMmmQ0']

  for video_title in file_list:
    print '\n###############'
    print video_title + '\n'

    scores_list = exp2(os.path.join('data_new', video_title + '.csv'))

    tex_filename = os.path.join(results_path, video_title + '.tex')
    figurename = os.path.join(figures_path, video_title)

    report.tex_report(tex_filename, video_title, scores_list)
    report.plot_mcc_bars(figurename, video_title, scores_list)
    report.plot_roc(figurename, video_title, scores_list)
