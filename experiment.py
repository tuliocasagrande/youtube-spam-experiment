#!/usr/bin/python
# This Python file uses the following encoding: utf-8

from classification import SingleClassification, DualClassification, SemiSupervisedClassification
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.grid_search import GridSearchCV
from sklearn.metrics  import accuracy_score, f1_score, matthews_corrcoef
from skll.metrics  import kappa
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import LinearSVC

def print_table_header(caption, label):
  s = '\\begin{table}[!htb]\n'
  s += '\\centering\n'
  s += '\\caption{{{0}}}\n'.format(caption)
  s += '\\label{{{0}}}\n'.format(label)
  s += '\\begin{tabular}{r|c|c|c|c|c|c|c|c|c|c}\n'
  s += '\\hline\\hline\n'
  s += 'Classifier & Acc (\\%) & SC (\\%) & BH (\\%) & F-medida & MCC & Kappa & TP & TN & FP & FN \\\\ \\hline\n'

  return s

def calculate_scores(y_true, y_pred):
  acc = accuracy_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)
  mcc = matthews_corrcoef(y_true, y_pred)
  kap = kappa(y_true, y_pred)

  tp = sum((y_true == y_pred) & (y_true == 1))
  tn = sum((y_true == y_pred) & (y_true == 0))
  fp = sum((y_true != y_pred) & (y_true == 0))
  fn = sum((y_true != y_pred) & (y_true == 1))

  try:
    sc = float(tp) / (tp + fn)
  except ZeroDivisionError:
    sc = 0

  try:
    bh = float(fp) / (fp + tn)
  except ZeroDivisionError:
    bh = 0

  scores = '{0:.2f} & {1:.2f} & {2:.2f} & '.format(acc * 100, sc * 100, bh * 100)
  scores += '{0:.3f} & {1:.3f} & {2:.3f} & '.format(f1, mcc, kap)
  scores += '{0} & {1} & {2} & {3} \\\\ \n'.format(tp, tn, fp, fn)

  return scores, f1

def print_table_footer():
  s = '\\hline\\hline\n'
  s += '\\end{tabular}\n'
  s += '\\end{table}\n'

  return s

def plot_figure(figure_name, scores_list):
  plt.figure()
  plt.title(figure_name)
  plt.xlabel('F-medida')

  performance = [f1 for f1, title, scores in scores_list]
  classifiers = tuple(title for f1, title, scores in scores_list)
  y_pos = np.arange(len(classifiers))
  plt.yticks(y_pos, classifiers)

  bar = plt.barh(y_pos, performance, align='center', alpha=0.4)
  best = performance[0]
  for i, p in enumerate(performance):
    if p == best:
      bar[i].set_color('r')

  plt.xticks(np.arange(0, 1.1, 0.1)) # guarantee an interval [0,1]
  plt.savefig(os.path.join(_figures_path, figure_name + '.png'), bbox_inches='tight')
  plt.savefig(os.path.join(_figures_path, figure_name + '.pdf'), bbox_inches='tight')

def exp1(file_prefix):

  with open(os.path.join(_results_path, os.path.basename(file_prefix)+'.tex'), 'w') as output_file:

    video_title = os.path.basename(file_prefix).split('-')[0]
    suffix_list = ['050', '075', '100', '125', '150']

    # Parameters for grid search
    range5 = [10.0 ** i for i in range(-5,5)]
    range_percent = [10 * i for i in range(1,11)]
    param_gamma = {'gamma': range5}
    param_C = {'C': range5}
    param_percentile = {'selectpercentile__percentile': range_percent}

    for suffix in suffix_list:
      scores_list = []
      caption = 'Resultados dos métodos de aprendizado de máquina para a base {0} do vídeo {1}.'.format(suffix, video_title)
      label = 'tab:{0}-{1}'.format(video_title, suffix)
      output_file.write(print_table_header(caption, label))
      filename = file_prefix + '-' + suffix + '.csv'

      # ========================= MultinomialNB 160/40 =========================
      pipeline = Pipeline([("selectpercentile", SelectPercentile(chi2)),
                         ("multinomialnb", MultinomialNB())])
      nb_grid = GridSearchCV(pipeline, param_percentile, cv=10, scoring='f1')

      y_true, y_pred = SingleClassification(filename, nb_grid).classify()
      scores, f1 = calculate_scores(y_true, y_pred)
      title = 'MultinomialNB'
      scores_list.append((f1, title, scores))

      # ========================== BernoulliNB 160/40 ==========================
      pipeline = Pipeline([("selectpercentile", SelectPercentile(chi2)),
                           ("bernoullinb", BernoulliNB())])
      nb_grid = GridSearchCV(pipeline, param_percentile, cv=10, scoring='f1')

      y_true, y_pred = SingleClassification(filename, nb_grid).classify()
      scores, f1 = calculate_scores(y_true, y_pred)
      title = 'BernoulliNB'
      scores_list.append((f1, title, scores))

      # =========================== LinearSVM 160/40 ===========================
      svm_grid = GridSearchCV(LinearSVC(), param_C, cv=10, scoring='f1')

      y_true, y_pred = SingleClassification(filename, svm_grid).classify()
      scores, f1 = calculate_scores(y_true, y_pred)
      title = 'SVM'
      scores_list.append((f1, title, scores))

      # ================= MultinomialNB + LinearSVM 160/20/20 ==================
      svm_grid = GridSearchCV(LinearSVC(), param_C, cv=10, scoring='f1')
      pipeline = Pipeline([("selectpercentile", SelectPercentile(chi2)),
                           ("multinomialnb", MultinomialNB())])
      nb_grid = GridSearchCV(pipeline, param_percentile, cv=10, scoring='f1')

      y_true, y_pred = DualClassification(filename, nb_grid, svm_grid).classify()
      scores, f1 = calculate_scores(y_true, y_pred)
      title = 'MultiNB & SVM'
      scores_list.append((f1, title, scores))

      y_true, y_pred = DualClassification(filename, nb_grid, svm_grid, 0.9).classify()
      scores, f1 = calculate_scores(y_true, y_pred)
      title = 'MultiNB & SVM (thresh 0.9)'
      scores_list.append((f1, title, scores))

      # ================= BernoulliNB + LinearSVM 160/20/20 ==================
      svm_grid = GridSearchCV(LinearSVC(), param_C, cv=10, scoring='f1')
      pipeline = Pipeline([("selectpercentile", SelectPercentile(chi2)),
                           ("bernoullinb", BernoulliNB())])
      nb_grid = GridSearchCV(pipeline, param_percentile, cv=10, scoring='f1')

      y_true, y_pred = DualClassification(filename, nb_grid, svm_grid).classify()
      scores, f1 = calculate_scores(y_true, y_pred)
      title = 'BernoNB & SVM'
      scores_list.append((f1, title, scores))

      y_true, y_pred = DualClassification(filename, nb_grid, svm_grid, 0.9).classify()
      scores, f1 = calculate_scores(y_true, y_pred)
      title = 'BernoNB & SVM (thresh 0.9)'
      scores_list.append((f1, title, scores))

      # ============== LabelPropagationRBF + LinearSVM 160/20/20 ===============
      svm_grid = GridSearchCV(LinearSVC(), param_C, cv=10, scoring='f1')
      lab_prop_grid = GridSearchCV(LabelPropagation(kernel='rbf'), param_gamma, cv=10, scoring='f1')

      y_true, y_pred = SemiSupervisedClassification(filename, lab_prop_grid, svm_grid).classify()
      scores, f1 = calculate_scores(y_true, y_pred)
      title = 'LabProp & SVM'
      scores_list.append((f1, title, scores))

      y_true, y_pred = SemiSupervisedClassification(filename, lab_prop_grid, svm_grid, 0.9).classify()
      scores, f1 = calculate_scores(y_true, y_pred)
      title = 'LabProp & SVM (thresh 0.9)'
      scores_list.append((f1, title, scores))

      # =============== LabelSpreadingRBF + LinearSVM 160/20/20 ================
      svm_grid = GridSearchCV(LinearSVC(), param_C, cv=10, scoring='f1')
      lab_spread_grid = GridSearchCV(LabelSpreading(kernel='rbf'), param_gamma, cv=10, scoring='f1')

      y_true, y_pred = SemiSupervisedClassification(filename, lab_spread_grid, svm_grid).classify()
      scores, f1 = calculate_scores(y_true, y_pred)
      title = 'LabSpread & SVM'
      scores_list.append((f1, title, scores))

      y_true, y_pred = SemiSupervisedClassification(filename, lab_spread_grid, svm_grid, 0.9).classify()
      scores, f1 = calculate_scores(y_true, y_pred)
      title = 'LabSpread & SVM (thresh 0.9)'
      scores_list.append((f1, title, scores))

      # ================================ SCORES ================================
      ordered_scores_list = sorted(scores_list, key=lambda scores: scores[0], reverse=True)

      plot_figure('{0}-{1}'.format(video_title, suffix), ordered_scores_list)

      for f1, title, scores in ordered_scores_list:
        output_file.write(title.replace('&','\&') + ' & ' + scores)
      output_file.write(print_table_footer())

if __name__ == "__main__":
  _results_path = os.path.join('exp1', 'results')
  _figures_path = os.path.join('exp1', 'figures')

  if not os.path.exists(_results_path):
    os.makedirs(_results_path)
  if not os.path.exists(_figures_path):
    os.makedirs(_figures_path)

  exp1(os.path.join('data', 'KatyPerry-CevxZvSJLk8'))
  exp1(os.path.join('data', 'PewDiePie-gRyPjRrjS34'))
  exp1(os.path.join('data', 'Psy-9bZkp7q19f0'))
