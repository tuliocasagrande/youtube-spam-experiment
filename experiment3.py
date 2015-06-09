#!/usr/bin/python
# This Python file uses the following encoding: utf-8

from classification import calculate_scores, SingleClassification, DualClassification, SemiSupervisedClassification
import os
from report import Report
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


def exp3(video_title, filename, output_report):

  # Parameters for grid search
  range5 = [10.0 ** i for i in range(-5,5)]
  range_percent = [10 * i for i in range(1,11)]
  param_gamma = {'gamma': range5}
  param_C = {'C': range5}
  param_C_gamma = {'C': range5, 'gamma': range5}
  param_percentile = {'selectpercentile__percentile': range_percent}

  output_report.new_table(video_title)

  # ========================= MultinomialNB 70%/30% ==========================
  clf_title = 'MultinomialNB'
  pipeline = Pipeline([("selectpercentile", SelectPercentile(chi2, percentile=70)),
                       ("multinomialnb", MultinomialNB())])
  y_true, y_pred = SingleClassification(filename, pipeline, train_percent=0.7).classify()
  output_report.append_scores(clf_title, calculate_scores(y_true, y_pred))

  # ========================== BernoulliNB 70%/30% ===========================
  clf_title = 'BernoulliNB'
  pipeline = Pipeline([("selectpercentile", SelectPercentile(chi2, percentile=70)),
                       ("bernoullinb", BernoulliNB())])
  y_true, y_pred = SingleClassification(filename, pipeline, train_percent=0.7).classify()
  output_report.append_scores(clf_title, calculate_scores(y_true, y_pred))

  # =========================== LinearSVM 70%/30% ============================
  clf_title = 'SVM Linear'
  grid = GridSearchCV(LinearSVC(), param_C, cv=10, scoring='f1')
  y_true, y_pred = SingleClassification(filename, grid, train_percent=0.7).classify()
  output_report.append_scores(clf_title, calculate_scores(y_true, y_pred))

  # =========================== SVM RBF 70%/30% ============================
  clf_title = 'SVM RBF'
  grid = GridSearchCV(SVC(kernel='rbf'), param_C_gamma, cv=10, scoring='f1')
  y_true, y_pred = SingleClassification(filename, grid, train_percent=0.7).classify()
  output_report.append_scores(clf_title, calculate_scores(y_true, y_pred))

  # =========================== SVM Poly 70%/30% ============================
  clf_title = 'SVM Poly'
  grid = GridSearchCV(SVC(kernel='poly'), param_C_gamma, cv=10, scoring='f1')
  y_true, y_pred = SingleClassification(filename, grid, train_percent=0.7).classify()
  output_report.append_scores(clf_title, calculate_scores(y_true, y_pred))

  # ======================= LogisticRegression 70%/30% =======================
  clf_title = 'Logistic'
  y_true, y_pred = SingleClassification(filename, LogisticRegression(), train_percent=0.7).classify()
  output_report.append_scores(clf_title, calculate_scores(y_true, y_pred))

  # ======================= DecisionTree 70%/30% =======================
  # scikit-learn uses an optimised version of the CART algorithm.
  clf_title = 'DecisionTree'
  y_true, y_pred = SingleClassification(filename, DecisionTreeClassifier(), train_percent=0.7).classify()
  output_report.append_scores(clf_title, calculate_scores(y_true, y_pred))

  # ======================= RandomForest 70%/30% =======================
  clf_title = 'RandomForest'
  y_true, y_pred = SingleClassification(filename, RandomForestClassifier(), train_percent=0.7).classify()
  output_report.append_scores(clf_title, calculate_scores(y_true, y_pred))

  # ======================= AdaBoost 70%/30% =======================
  clf_title = 'AdaBoost'
  y_true, y_pred = SingleClassification(filename, AdaBoostClassifier(), train_percent=0.7).classify()
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
