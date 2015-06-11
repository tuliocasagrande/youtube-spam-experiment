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


def exp3(filename):

  # Parameters for grid search
  range5 = [10.0 ** i for i in range(-5,5)]
  param_gamma = {'gamma': range5}
  param_C = {'C': range5}
  param_C_gamma = {'C': range5, 'gamma': range5}
  param_nestimators = {'n_estimators': range(10,101,10)}
  mcc = make_scorer(matthews_corrcoef)

  scores_list = []

  config = [('MultinomialNB', MultinomialNB()),
            ('BernoulliNB', BernoulliNB()),
            ('GaussianNB', GaussianNB()),
            ('SVM Linear', GridSearchCV(LinearSVC(), param_C, cv=10, scoring=mcc)),
            ('SVM RBF', GridSearchCV(SVC(kernel='rbf'), param_C_gamma, cv=10, scoring=mcc)),
            ('SVM Poly', GridSearchCV(SVC(kernel='poly'), param_C_gamma, cv=10, scoring=mcc)),
            ('Logistic', GridSearchCV(LogisticRegression(), param_C, cv=10, scoring=mcc)),
            ('DecisionTree', DecisionTreeClassifier()),
            ('RandomForest', GridSearchCV(RandomForestClassifier(), param_nestimators, cv=10, scoring=mcc)),
            ('AdaBoost', AdaBoostClassifier()),
            ('1-NN', KNeighborsClassifier(n_neighbors=1)),
            ('3-NN', KNeighborsClassifier(n_neighbors=3)),
            ('5-NN', KNeighborsClassifier(n_neighbors=5))]

  single_classification = SingleClassification(filename, train_percent=0.7)
  for clf_title, clf in config:
    y_true, y_pred = single_classification.classify(clf)
    scores_list.append((clf_title, calculate_scores(y_true, y_pred)))

  scores_list.sort(key=lambda scores: (scores[1]['mcc'], scores[1]['f1']), reverse=True)
  return scores_list


if __name__ == "__main__":
  results_path = os.path.join('exp3', 'results')
  figures_path = os.path.join('exp3', 'figures')

  if not os.path.exists(results_path):
    os.makedirs(results_path)
  if not os.path.exists(figures_path):
    os.makedirs(figures_path)

  file_list = ['01-PSY-9bZkp7q19f0',
               '04-KatyPerry-CevxZvSJLk8',
               '07-LMFAO-KQ6zr6kCPj8',
               '08-Eminem-uelHwf8o7_U',
               '09-Shakira-pRpeEdMmmQ0']

  csv_filename = os.path.join(results_path, 'results_mcc.csv')
  report.csv_init_header(csv_filename)

  for video_title in file_list:
    scores_list = exp3(os.path.join('data_new', video_title + '.csv'))

    tex_filename = os.path.join(results_path, video_title + '.tex')
    figurename = os.path.join(figures_path, video_title)

    report.tex_report(tex_filename, video_title, scores_list)
    report.csv_report(csv_filename, video_title, scores_list)
    report.plot_figure(figurename, video_title, scores_list)
