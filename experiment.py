#!/usr/bin/python
# This Python file uses the following encoding: utf-8

from classification import SingleClassification, DualClassification, SemiSupervisedClassification
import csv
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

def print_scores(y_true, y_pred):
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

  s = '{0:.2f} & {1:.2f} & {2:.2f} & '.format(acc * 100, sc * 100, bh * 100)
  s += '{0:.3f} & {1:.3f} & {2:.3f} & '.format(f1, mcc, kap)
  s += '{0} & {1} & {2} & {3} \\\\ \n'.format(tp, tn, fp, fn)

  return s

def print_table_footer():
  s = '\\hline\\hline\n'
  s += '\\end{tabular}\n'
  s += '\\end{table}\n'

  return s

def execute(file_prefix):

  with open('results/'+os.path.basename(file_prefix)+'.tex', 'w') as output_file:

    video_title = os.path.basename(file_prefix).split('-')[0]
    suffix_list = ['-050', '-075', '-100', '-125', '-150']

    captions = ['Avaliando MultinomialNB com 160 amostras de treino e 40 amostras de teste para o vídeo {0}.'.format(video_title),
                'Avaliando BernoulliNB com 160 amostras de treino e 40 amostras de teste para o vídeo {0}.'.format(video_title),
                'Avaliando SVM Linear com 160 amostras de treino e 40 amostras de teste para o vídeo {0}.'.format(video_title),
                'Avaliando SVM Linear com 160 amostras rotuladas manualmente, 20 amostras rotuladas pelo MultinomialNB e 20 amostras de teste para o vídeo {0}.'.format(video_title),
                'Avaliando SVM Linear com 160 amostras rotuladas manualmente, 20 amostras rotuladas pelo LabelPropagation RBF e 20 amostras de teste para o vídeo {0}.'.format(video_title),
                'Avaliando SVM Linear com 160 amostras rotuladas manualmente, 20 amostras rotuladas pelo LabelSpreading RBF e 20 amostras de teste para o vídeo {0}.'.format(video_title)]

    labels = ['tab:multinomial-nb-{0}'.format(video_title),
              'tab:bernoulli-nb-{0}'.format(video_title),
              'tab:linear-svm-{0}'.format(video_title),
              'tab:multinomial-nb-linear-svm-{0}'.format(video_title),
              'tab:labprop-rbf-linear-svm-{0}'.format(video_title),
              'tab:labspread-rbf-linear-svm-{0}'.format(video_title)]

    # Parameters for grid search
    range5 = [10.0 ** i for i in range(-5,5)]
    range_percent = [10 * i for i in range(1,11)]
    param_gamma = {'gamma': range5}
    param_C = {'C': range5}
    param_percentile = {'selectpercentile__percentile': range_percent}

    # ========================== MultinomialNB 160/40 ==========================
    output_file.write(print_table_header(captions[0], labels[0]))

    pipeline = Pipeline([("selectpercentile", SelectPercentile(chi2)),
                         ("multinomialnb", MultinomialNB())])
    nb_grid = GridSearchCV(pipeline, param_percentile, cv=10, scoring='f1')

    for each in suffix_list:
      filename = file_prefix + each + '.csv'
      y_true, y_pred = SingleClassification(filename, nb_grid).classify()
      output_file.write('MultinomialNB{0} & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

    output_file.write(print_table_footer())

    # =========================== BernoulliNB 160/40 ===========================
    output_file.write(print_table_header(captions[1], labels[1]))

    pipeline = Pipeline([("selectpercentile", SelectPercentile(chi2)),
                         ("bernoullinb", BernoulliNB())])
    nb_grid = GridSearchCV(pipeline, param_percentile, cv=10, scoring='f1')

    for each in suffix_list:
      filename = file_prefix + each + '.csv'
      y_true, y_pred = SingleClassification(filename, nb_grid).classify()
      output_file.write('BernoulliNB{0} & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

    output_file.write(print_table_footer())

    # ============================ LinearSVM 160/40 ============================
    output_file.write(print_table_header(captions[2], labels[2]))
    svm_grid = GridSearchCV(LinearSVC(), param_C, cv=10, scoring='f1')

    for each in suffix_list:
      filename = file_prefix + each + '.csv'
      y_true, y_pred = SingleClassification(filename, svm_grid).classify()
      output_file.write('SVM{0} & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

    output_file.write(print_table_footer())

    # ================= MultinomialNB + LinearSVM 160/20/20 ====================
    output_file.write(print_table_header(captions[3], labels[3]))
    svm_grid = GridSearchCV(LinearSVC(), param_C, cv=10, scoring='f1')
    pipeline = Pipeline([("selectpercentile", SelectPercentile(chi2)),
                         ("multinomialnb", MultinomialNB())])
    nb_grid = GridSearchCV(pipeline, param_percentile, cv=10, scoring='f1')


    for each in suffix_list:
      filename = file_prefix + each + '.csv'
      y_true, y_pred = DualClassification(filename, nb_grid, svm_grid).classify()
      output_file.write('MultiNB \& SVM{0} & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

      y_true, y_pred = DualClassification(filename, nb_grid, svm_grid, 0.7).classify()
      output_file.write('MultiNB \& SVM{0} (thr 0.7) & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

      y_true, y_pred = DualClassification(filename, nb_grid, svm_grid, 0.8).classify()
      output_file.write('MultiNB \& SVM{0} (thr 0.8) & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

      y_true, y_pred = DualClassification(filename, nb_grid, svm_grid, 0.9).classify()
      output_file.write('MultiNB \& SVM{0} (thr 0.9) & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

    output_file.write(print_table_footer())

    # ============== LabelPropagationRBF + LinearSVM 160/20/20 =================
    output_file.write(print_table_header(captions[4], labels[4]))
    svm_grid = GridSearchCV(LinearSVC(), param_C, cv=10, scoring='f1')
    lab_prop_grid = GridSearchCV(LabelPropagation(kernel='rbf'), param_gamma, cv=10, scoring='f1')

    for each in suffix_list:
      filename = file_prefix + each + '.csv'
      y_true, y_pred = SemiSupervisedClassification(filename, lab_prop_grid, svm_grid).classify()
      output_file.write('LabProp \& SVM{0} & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

      y_true, y_pred = SemiSupervisedClassification(filename, lab_prop_grid, svm_grid, 0.7).classify()
      output_file.write('LabProp \& SVM{0} (thr 0.7) & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

      y_true, y_pred = SemiSupervisedClassification(filename, lab_prop_grid, svm_grid, 0.8).classify()
      output_file.write('LabProp \& SVM{0} (thr 0.8) & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

      y_true, y_pred = SemiSupervisedClassification(filename, lab_prop_grid, svm_grid, 0.9).classify()
      output_file.write('LabProp \& SVM{0} (thr 0.9) & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

    output_file.write(print_table_footer())

    # =============== LabelSpreadingRBF + LinearSVM 160/20/20 ==================
    output_file.write(print_table_header(captions[5], labels[5]))
    svm_grid = GridSearchCV(LinearSVC(), param_C, cv=10, scoring='f1')
    lab_spread_grid = GridSearchCV(LabelSpreading(kernel='rbf'), param_gamma, cv=10, scoring='f1')

    for each in suffix_list:
      filename = file_prefix + each + '.csv'
      y_true, y_pred = SemiSupervisedClassification(filename, lab_spread_grid, svm_grid).classify()
      output_file.write('LabSpread \& SVM{0} & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

      y_true, y_pred = SemiSupervisedClassification(filename, lab_spread_grid, svm_grid, 0.7).classify()
      output_file.write('LabSpread \& SVM{0} (thr 0.7) & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

      y_true, y_pred = SemiSupervisedClassification(filename, lab_spread_grid, svm_grid, 0.8).classify()
      output_file.write('LabSpread \& SVM{0} (thr 0.8) & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

      y_true, y_pred = SemiSupervisedClassification(filename, lab_spread_grid, svm_grid, 0.9).classify()
      output_file.write('LabSpread \& SVM{0} (thr 0.9) & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

    output_file.write(print_table_footer())


if __name__ == "__main__":
  if not os.path.exists('results'):
    os.makedirs('results')

  execute('data/KatyPerry-CevxZvSJLk8')
  execute('data/PewDiePie-gRyPjRrjS34')
  execute('data/Psy-9bZkp7q19f0')
