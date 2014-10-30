#!/usr/bin/python
# This Python file uses the following encoding: utf-8

import csv, sys
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import LinearSVC
import numpy as np

class Classifier:
  """Store the classifier and its achieved results"""

  def __init__(self, algorithm, param_grid, cv=5):
    self.tp = []
    self.tn = []
    self.fp = []
    self.fn = []
    self.grid = GridSearchCV(algorithm(), param_grid, cv=cv)

  def fit(self, X, y):
    """
      Perform the GridSearch with cross-validation and save the trained classifier.
    """
    self.grid.fit(X, y)
    self.clf = self.grid.best_estimator_

  def predict(self, unlabeled_X, y=None):
    predicted_y = self.clf.predict(unlabeled_X)

    if np.any(y):
      self.tp.append(sum((y == predicted_y) & (y == 1)))
      self.tn.append(sum((y == predicted_y) & (y == 0)))
      self.fp.append(sum((y != predicted_y) & (y == 0)))
      self.fn.append(sum((y != predicted_y) & (y == 1)))

    return predicted_y

  def report(self):

    tp = np.array(self.tp, dtype='float64'); tn = np.array(self.tn, dtype='float64')
    fp = np.array(self.fp, dtype='float64'); fn = np.array(self.fn, dtype='float64')

    self.acc = (tp + tn) / (tp + tn + fp + fn)
    self.sc = tp / (tp + fn)
    self.bh = fp / (fp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    self.f1 = 2 * (precision * recall) / (precision + recall)
    self.mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    # Printing scores
    # x +/- 2sigma (approximately a 95% confidence interval)
    sys.stdout.write('{0} & '.format(type(self.clf).__name__))
    sys.stdout.write('{0:.2f} $\pm$ {1:.2f} & '.format(np.nanmean(self.acc) * 100, np.nanstd(self.acc) * 2))
    sys.stdout.write('{0:.2f} $\pm$ {1:.2f} & '.format(np.nanmean(self.sc) * 100, np.nanstd(self.sc) * 2))
    sys.stdout.write('{0:.2f} $\pm$ {1:.2f} & '.format(np.nanmean(self.bh) * 100, np.nanstd(self.bh) * 2))
    sys.stdout.write('{0:.3f} $\pm$ {1:.3f} & '.format(np.nanmean(self.f1) * 100, np.nanstd(self.f1) * 2))
    sys.stdout.write('{0:.3f} $\pm$ {1:.3f} \\\\\n'.format(np.nanmean(self.mcc) * 100, np.nanstd(self.mcc) * 2))


def classify(filename):

  contents = []
  classes = []

  # Reading and parsing CSV file
  with open(filename, 'rb') as csvfile:
    reader = csv.reader(csvfile)
    reader.next() # Skiping the header

    for row in reader:
      contents.append(row[1])
      classes.append(int(row[2]))

  # Preparing data
  contents = np.asarray(contents)
  classes = np.asarray(classes)

  # Array of permutations
  permut = np.random.permutation(200)

  # Semi-supervised training, 100/50/50
  caption = 'Evaluating semi-supervised training with 100 training samples, 50 semi-supervised samples and 50 testing samples when using LabelPropagation as the semi-supervised algorithm.'
  label = 'tab:label-propagation'
  print_table_header(caption, label)
  semi_supervised_training(LabelPropagation, contents, classes, permut)
  print_table_footer()

  caption = 'Evaluating semi-supervised training with 100 training samples, 50 semi-supervised samples and 50 testing samples when using LabelSpreading as the semi-supervised algorithm (more robust to noise).'
  label = 'tab:label-spreading'
  print_table_header(caption, label)
  semi_supervised_training(LabelSpreading, contents, classes, permut)
  print_table_footer()

  # Linear SVM, 100/100
  caption = 'Evaluating supervised training with 100 training samples and 100 testing samples when using LinearSVM as the supervised algorithm.'
  label = 'tab:linear-svm'
  print_table_header(caption, label)
  supervised_training(contents, classes, permut)
  print_table_footer()

def print_table_header(caption, label):
  print '\\begin{table}[!htb]'
  print '\\footnotesize'
  print '\\centering'
  print '\\caption{{{0}}}'.format(caption)
  print '\\label{{{0}}}'.format(label)
  print '\\begin{tabular}{r|c|c|c|c|c} \\hline\\hline'
  print 'Classifier & Acc (\\%) & SC (\\%) & BH (\\%) & F-medida & MCC \\\\ \\hline'

def print_table_footer():
  print '\\hline\\hline'
  print '\\end{tabular}'
  print '\\end{table}'

# Evaluating semi-supervised training with an adaptated leave-one-out:
# 1- train with 100 labeled + 50 unlabeled comments;
# 2- test with the remaning 50 comments;
# 3- rotate the dataset by 1 comment and repeat 200 times
def semi_supervised_training(semi_superv_algorithm, contents, classes, permut):

  range5 = list((10.0**i) for i in range(-5,5))
  param_gamma = {'gamma': range5}
  param_C = {'C': range5}

  semi_supervised_clf = Classifier(semi_superv_algorithm, param_gamma, 5)
  svm_clf = Classifier(LinearSVC, param_C, 5)

  unlabeled_data = np.repeat(-1, 50)

  for i in xrange(0,200):
    train_index = permut[:100]
    semi_supervised_index = permut[100:150]
    test_index = permut[150:]

    # Preparing bag of words
    vectorizer = CountVectorizer(min_df=1)

    # Merging training and semi-supervised data
    X = vectorizer.fit_transform(
        np.concatenate([ contents[train_index],
                            contents[semi_supervised_index] ]))

    # Merging training labels and semi-supervised labels (-1)
    y = np.concatenate([classes[train_index], unlabeled_data])

    # Evaluating semi-supervised training
    semi_supervised_clf.fit(X, y)
    semi_supervised_X = vectorizer.transform(contents[semi_supervised_index])
    semi_supervised_y = semi_supervised_clf.predict(
                            semi_supervised_X, classes[semi_supervised_index])

    # Merging training labels and predicted labels
    y = np.concatenate([classes[train_index], semi_supervised_y])

    # Evaluating SVM training
    svm_clf.fit(X, y)
    test_X = vectorizer.transform(contents[test_index])
    test_y = svm_clf.predict(test_X, classes[test_index])

    # Rotating dataset
    permut = np.delete(np.append(permut, permut[0]), 0)

  semi_supervised_clf.report()
  svm_clf.report()

# Evaluating supervised training with an adaptated leave-one-out:
# 1- train with 100 labeled comments;
# 2- test with the remaning 100 comments;
# 3- rotate the dataset by 1 comment and repeat 200 times
def supervised_training(contents, classes, permut):

  range5 = list((10.0**i) for i in range(-5,5))
  param_C = {'C': range5}
  svm_clf = Classifier(LinearSVC, param_C, 5)

  unlabeled_data = np.repeat(-1, 50)

  for i in xrange(0,200):
    train_index = permut[:100]
    test_index = permut[100:150]

    # Preparing bag of words
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(contents[train_index])
    y = classes[train_index]

    # Evaluating SVM training
    svm_clf.fit(X, y)
    test_X = vectorizer.transform(contents[test_index])
    test_y = svm_clf.predict(test_X, classes[test_index])

    # Rotating dataset
    permut = np.delete(np.append(permut, permut[0]), 0)

  svm_clf.report()

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print 'Usage: {0} filename'.format(sys.argv[0])
  else:
    classify(sys.argv[1])
