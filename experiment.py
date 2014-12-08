#!/usr/bin/python
# This Python file uses the following encoding: utf-8

import csv
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.grid_search import GridSearchCV
from sklearn.metrics  import accuracy_score, f1_score, matthews_corrcoef
from skll.metrics  import kappa
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import LinearSVC

def read_file(filename):

  contents = []
  classes = []

  # Reading and parsing CSV file
  with open(filename, 'rb') as csvfile:
    reader = csv.reader(csvfile)
    reader.next() # Skiping the header

    for row in reader:
      contents.append(row[3])
      classes.append(int(row[4]))

  contents = np.asarray(contents)
  classes = np.asarray(classes)

  return contents, classes

# Evaluating supervised training:
# 1- train with the 160 oldest labeled comments (80%)
# 2- test with the remaning 40 comments (20%)
def experiment_one_clf(filename, clf):

  # Reading file
  contents, classes = read_file(filename)

  # Just to keep it dynamically, 160 = 80% of 200 total samples
  train_index = int(len(classes) * 0.8)

  # The dataset is ordered by date (first = oldest)
  X_train = contents[:train_index]
  y_train = classes[:train_index]

  X_test = contents[train_index:]
  y_test = classes[train_index:]

  # Preparing bag of words
  vectorizer = CountVectorizer(min_df=1)
  bow_X = vectorizer.fit_transform(X_train)

  # Evaluating supervised training
  clf.fit(bow_X, y_train)
  bow_X = vectorizer.transform(X_test)
  y_pred = clf.predict(bow_X)

  return y_test, y_pred

# Evaluating dual supervised training:
# 1- train with the 160 oldest labeled comments (80%)
# 2- predict half the remaining comments with NaiveBayes (10%)
# 3- test svm with the other half (10%)
def experiment_dual_clf(filename, nb_clf, svm_clf):

  # Reading file
  contents, classes = read_file(filename)

  # Just to keep it dynamically, 160 = 80% of 200 total samples
  train_index = int(len(classes) * 0.8)
  semisupervised_index = int(len(classes) * 0.9)

  # The dataset is ordered by date (first = oldest)
  X_train = contents[:train_index]
  y_train = classes[:train_index]

  X_semisupervised = contents[train_index:semisupervised_index]
  y_semisupervised = classes[train_index:semisupervised_index]

  X_test = contents[semisupervised_index:]
  y_test = classes[semisupervised_index:]

  # ===================== Evaluating naive bayes training ======================

  # Preparing bag of words
  vectorizer = CountVectorizer(min_df=1)
  bow_X = vectorizer.fit_transform(X_train)

  nb_clf.fit(bow_X, y_train)
  bow_X = vectorizer.transform(X_semisupervised)
  y_pred_nb = nb_clf.predict(bow_X)

  # ========================= Evaluating svm training ==========================

  # Preparing bag of words
  vectorizer = CountVectorizer(min_df=1)
  bow_X = vectorizer.fit_transform(np.concatenate([X_train, X_semisupervised]))
  y = np.concatenate([y_train, y_pred_nb])

  svm_clf.fit(bow_X, y)
  bow_X = vectorizer.transform(X_test)
  y_pred_svm = svm_clf.predict(bow_X)

  # ======================== Results of both trainings =========================

  # The scores should consider all errors
  y_true = np.concatenate([y_semisupervised, y_test])
  y_pred = np.concatenate([y_pred_nb, y_pred_svm])

  return y_true, y_pred

# Evaluating semi-supervised training:
# 1- train with the 160 oldest labeled comments (80%)
# 2- predict half the remaining comments with semi-supervised training (10%)
# 3- test svm with the other half (10%)
def experiment_semisupervised(filename, ss_clf, svm_clf):

  # Reading file
  contents, classes = read_file(filename)

  # Just to keep it dynamically, 160 = 80% of 200 total samples
  train_index = int(len(classes) * 0.8)
  semisupervised_index = int(len(classes) * 0.9)

  # The dataset is ordered by date (first = oldest)
  X_train = contents[:train_index]
  y_train = classes[:train_index]

  X_semisupervised = contents[train_index:semisupervised_index]
  y_semisupervised = classes[train_index:semisupervised_index]

  X_test = contents[semisupervised_index:]
  y_test = classes[semisupervised_index:]

  # Semi-supervised labels (-1)
  unlabeled_data = np.repeat(-1, semisupervised_index - train_index)

  # ==================== Evaluating semi-supervised training ===================

  # Preparing bag of words
  vectorizer = CountVectorizer(min_df=1)

  # Merging training and semi-supervised data
  bow_X = vectorizer.fit_transform(np.concatenate([X_train, X_semisupervised]))
  y = np.concatenate([y_train, unlabeled_data])

  ss_clf.fit(bow_X, y)
  bow_X = vectorizer.transform(X_semisupervised)
  y_pred_ss = ss_clf.predict(bow_X)

  # ========================== Evaluating svm training =========================

  # Preparing bag of words
  vectorizer = CountVectorizer(min_df=1)

  # Merging training and semi-supervised data (now labeled!)
  bow_X = vectorizer.fit_transform(np.concatenate([X_train, X_semisupervised]))
  y = np.concatenate([y_train, y_pred_ss])

  svm_clf.fit(bow_X, y)
  bow_X = vectorizer.transform(X_test)
  y_pred_svm = svm_clf.predict(bow_X)

  # ======================== Results of both trainings =========================

  # The scores should consider all errors
  y_true = np.concatenate([y_semisupervised, y_test])
  y_pred = np.concatenate([y_pred_ss, y_pred_svm])

  return y_true, y_pred


# Evaluating dual supervised training:
# 1- train with the 160 oldest labeled comments (80%)
# 2- predict half the remaining comments with NaiveBayes (10%)
# 3- test svm with the other half (10%)
def experiment_dual_clf_with_thresh(filename, nb_clf, svm_clf, threshold = 0.8):

  # Reading file
  contents, classes = read_file(filename)

  # Just to keep it dynamically, 160 = 80% of 200 total samples
  train_index = int(len(classes) * 0.8)
  semisupervised_index = int(len(classes) * 0.9)

  # The dataset is ordered by date (first = oldest)
  X_train = contents[:train_index]
  y_train = classes[:train_index]

  X_semisupervised = contents[train_index:semisupervised_index]
  y_semisupervised = classes[train_index:semisupervised_index]

  X_test = contents[semisupervised_index:]
  y_test = classes[semisupervised_index:]

  # ===================== Evaluating naive bayes training ======================

  # Preparing bag of words
  vectorizer = CountVectorizer(min_df=1)
  bow_X = vectorizer.fit_transform(X_train)

  nb_clf.fit(bow_X, y_train)
  bow_X = vectorizer.transform(X_semisupervised)
  y_pred_nb = nb_clf.predict(bow_X)

  y_proba_nb = nb_clf.predict_proba(bow_X)
  above_threshold = [each[y_pred_nb[i]] >= threshold for i, each in enumerate(y_proba_nb)]
  above_threshold = np.asarray(above_threshold)

  # ========================= Evaluating svm training ==========================

  # Preparing bag of words
  vectorizer = CountVectorizer(min_df=1)

  # Merging training and naive bayes above threshold results data
  bow_X = vectorizer.fit_transform(np.concatenate([X_train, X_semisupervised[above_threshold]]))
  y = np.concatenate([y_train, y_pred_nb[above_threshold]])

  svm_clf.fit(bow_X, y)
  bow_X = vectorizer.transform(np.concatenate([X_semisupervised[~above_threshold], X_test]))
  y_pred_svm = svm_clf.predict(bow_X)

  # ======================== Results of both trainings =========================

  # The scores should consider all errors
  y_true = np.concatenate([y_semisupervised, y_test])
  y_pred = np.concatenate([y_pred_nb[above_threshold], y_pred_svm])

  return y_true, y_pred

# Evaluating semi-supervised training:
# 1- train with the 160 oldest labeled comments (80%)
# 2- predict half the remaining comments with semi-supervised training (10%)
# 3- test svm with the other half (10%)
def experiment_semisupervised_with_thresh(filename, ss_clf, svm_clf, threshold = 0.8):

  # Reading file
  contents, classes = read_file(filename)

  # Just to keep it dynamically, 160 = 80% of 200 total samples
  train_index = int(len(classes) * 0.8)
  semisupervised_index = int(len(classes) * 0.9)

  # The dataset is ordered by date (first = oldest)
  X_train = contents[:train_index]
  y_train = classes[:train_index]

  X_semisupervised = contents[train_index:semisupervised_index]
  y_semisupervised = classes[train_index:semisupervised_index]

  X_test = contents[semisupervised_index:]
  y_test = classes[semisupervised_index:]

  # Semi-supervised labels (-1)
  unlabeled_data = np.repeat(-1, semisupervised_index - train_index)

  # ==================== Evaluating semi-supervised training ===================

  # Preparing bag of words
  vectorizer = CountVectorizer(min_df=1)

  # Merging training and semi-supervised data
  bow_X = vectorizer.fit_transform(np.concatenate([X_train, X_semisupervised]))
  y = np.concatenate([y_train, unlabeled_data])

  ss_clf.fit(bow_X, y)
  bow_X = vectorizer.transform(X_semisupervised)
  y_pred_ss = ss_clf.predict(bow_X)

  y_proba_ss = ss_clf.predict_proba(bow_X)
  above_threshold = [each[y_pred_ss[i]] >= threshold for i, each in enumerate(y_proba_ss)]
  above_threshold = np.asarray(above_threshold)

  # ========================== Evaluating svm training =========================

  # Preparing bag of words
  vectorizer = CountVectorizer(min_df=1)

  # Merging training and semi-supervised data (above threshold)
  bow_X = vectorizer.fit_transform(np.concatenate([X_train, X_semisupervised[above_threshold]]))
  y = np.concatenate([y_train, y_pred_ss[above_threshold]])

  svm_clf.fit(bow_X, y)
  bow_X = vectorizer.transform(np.concatenate([X_semisupervised[~above_threshold], X_test]))
  y_pred_svm = svm_clf.predict(bow_X)

  # ======================== Results of both trainings =========================

  # The scores should consider all errors
  y_true = np.concatenate([y_semisupervised, y_test])
  y_pred = np.concatenate([y_pred_ss[above_threshold], y_pred_svm])

  return y_true, y_pred


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
    range5 = list((10.0**i) for i in range(-5,5))
    param_gamma = {'gamma': range5}
    param_C = {'C': range5}
    # param_k = {'n_neighbors': [1,3,5,7]}

    # ========================== MultinomialNB 160/40 ==========================
    output_file.write(print_table_header(captions[0], labels[0]))
    nb_clf = MultinomialNB()

    for each in suffix_list:
      y_true, y_pred = experiment_one_clf(file_prefix + each + '.csv', nb_clf)
      output_file.write('MultinomialNB{0} & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

    output_file.write(print_table_footer())

    # =========================== BernoulliNB 160/40 ===========================
    output_file.write(print_table_header(captions[1], labels[1]))
    nb_clf = BernoulliNB()

    for each in suffix_list:
      y_true, y_pred = experiment_one_clf(file_prefix + each + '.csv', nb_clf)
      output_file.write('BernoulliNB{0} & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

    output_file.write(print_table_footer())

    # ============================ LinearSVM 160/40 ============================
    output_file.write(print_table_header(captions[2], labels[2]))
    svm_grid = GridSearchCV(LinearSVC(), param_C, cv=10, scoring='f1')

    for each in suffix_list:
      y_true, y_pred = experiment_one_clf(file_prefix + each + '.csv', svm_grid)
      output_file.write('SVM{0} & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

    output_file.write(print_table_footer())

    # ================= MultinomialNB + LinearSVM 160/20/20 ====================
    output_file.write(print_table_header(captions[3], labels[3]))
    svm_grid = GridSearchCV(LinearSVC(), param_C, cv=10, scoring='f1')
    nb_clf = MultinomialNB()

    for each in suffix_list:
      y_true, y_pred = experiment_dual_clf(file_prefix + each + '.csv', nb_clf, svm_grid)
      output_file.write('MultiNB \& SVM{0} & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

      y_true, y_pred = experiment_dual_clf_with_thresh(file_prefix + each + '.csv', nb_clf, svm_grid, 0.7)
      output_file.write('MultiNB \& SVM{0} (thr 0.7) & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

      y_true, y_pred = experiment_dual_clf_with_thresh(file_prefix + each + '.csv', nb_clf, svm_grid)
      output_file.write('MultiNB \& SVM{0} (thr 0.8) & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

      y_true, y_pred = experiment_dual_clf_with_thresh(file_prefix + each + '.csv', nb_clf, svm_grid, 0.9)
      output_file.write('MultiNB \& SVM{0} (thr 0.9) & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

    output_file.write(print_table_footer())

    # ============== LabelPropagationRBF + LinearSVM 160/20/20 =================
    output_file.write(print_table_header(captions[4], labels[4]))
    svm_grid = GridSearchCV(LinearSVC(), param_C, cv=10, scoring='f1')
    lab_prop_grid = GridSearchCV(LabelPropagation(kernel='rbf'), param_gamma, cv=10, scoring='f1')

    for each in suffix_list:
      y_true, y_pred = experiment_semisupervised(file_prefix + each + '.csv', lab_prop_grid, svm_grid)
      output_file.write('LabProp \& SVM{0} & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

      y_true, y_pred = experiment_semisupervised_with_thresh(file_prefix + each + '.csv', lab_prop_grid, svm_grid, 0.7)
      output_file.write('LabProp \& SVM{0} (thr 0.7) & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

      y_true, y_pred = experiment_semisupervised_with_thresh(file_prefix + each + '.csv', lab_prop_grid, svm_grid)
      output_file.write('LabProp \& SVM{0} (thr 0.8) & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

      y_true, y_pred = experiment_semisupervised_with_thresh(file_prefix + each + '.csv', lab_prop_grid, svm_grid, 0.9)
      output_file.write('LabProp \& SVM{0} (thr 0.9) & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

    output_file.write(print_table_footer())

    # =============== LabelSpreadingRBF + LinearSVM 160/20/20 ==================
    output_file.write(print_table_header(captions[5], labels[5]))
    svm_grid = GridSearchCV(LinearSVC(), param_C, cv=10, scoring='f1')
    lab_spread_grid = GridSearchCV(LabelSpreading(kernel='rbf'), param_gamma, cv=10, scoring='f1')

    for each in suffix_list:
      y_true, y_pred = experiment_semisupervised(file_prefix + each + '.csv', lab_spread_grid, svm_grid)
      output_file.write('LabSpread \& SVM{0} & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

      y_true, y_pred = experiment_semisupervised_with_thresh(file_prefix + each + '.csv', lab_spread_grid, svm_grid, 0.7)
      output_file.write('LabSpread \& SVM{0} (thr 0.7) & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

      y_true, y_pred = experiment_semisupervised_with_thresh(file_prefix + each + '.csv', lab_spread_grid, svm_grid)
      output_file.write('LabSpread \& SVM{0} (thr 0.8) & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

      y_true, y_pred = experiment_semisupervised_with_thresh(file_prefix + each + '.csv', lab_spread_grid, svm_grid, 0.9)
      output_file.write('LabSpread \& SVM{0} (thr 0.9) & '.format(each))
      output_file.write(print_scores(y_true, y_pred))

    output_file.write(print_table_footer())


if __name__ == "__main__":
  if not os.path.exists('results'):
    os.makedirs('results')

  execute('data/KatyPerry-CevxZvSJLk8')
  execute('data/PewDiePie-gRyPjRrjS34')
  execute('data/Psy-9bZkp7q19f0')
