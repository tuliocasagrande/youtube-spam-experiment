#!/usr/bin/python
# This Python file uses the following encoding: utf-8

from classification import calculate_scores, SingleClassification, DualClassification, SemiSupervisedClassification
import os
from report import Report
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


def exp3(video_title, filename, output_report):

  # Parameters for grid search
  range5 = [10.0 ** i for i in range(-5,5)]
  param_gamma = {'gamma': range5}
  param_C = {'C': range5}
  param_C_gamma = {'C': range5, 'gamma': range5}
  param_nestimators = {'n_estimators': range(10,101,10)}
  mcc = make_scorer(matthews_corrcoef)

  output_report.new_table(video_title)

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

  for clf_title, clf in config:
    y_true, y_pred = SingleClassification(filename, clf, train_percent=0.7).classify()
    output_report.append_scores(clf_title, calculate_scores(y_true, y_pred))


  # ================================ SCORES ================================
  output_report.print_scores()
  output_report.print_table_footer()
  output_report.plot_figure()


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

  for f in file_list:
    with open(os.path.join(results_path, f + '.tex'), 'w') as output_file:
      exp3(f, os.path.join('data_new', f+'.csv'), Report(output_file, figures_path))
