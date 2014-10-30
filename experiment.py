#!/usr/bin/python
# This Python file uses the following encoding: utf-8

from classification import semi_supervised_training, supervised_training
import csv, sys
import numpy as np
from os.path import splitext, basename
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

def experiment(path):
  video_name = splitext(basename(path))[0]

  contents = []
  classes = []

  # Reading and parsing CSV file
  with open(path, 'rb') as csvfile:
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
  label = 'tab:label-propagation-{0}'.format(video_name)
  print_table_header(caption, label)
  semi_supervised_training(LabelPropagation, contents, classes, permut)
  print_table_footer()

  caption = 'Evaluating semi-supervised training with 100 training samples, 50 semi-supervised samples and 50 testing samples when using LabelSpreading as the semi-supervised algorithm (more robust to noise).'
  label = 'tab:label-spreading-{0}'.format(video_name)
  print_table_header(caption, label)
  semi_supervised_training(LabelSpreading, contents, classes, permut)
  print_table_footer()

  # Linear SVM, 100/100
  caption = 'Evaluating supervised training with 100 training samples and 100 testing samples when using LinearSVM as the supervised algorithm.'
  label = 'tab:linear-svm-{0}'.format(video_name)
  print_table_header(caption, label)
  supervised_training(contents, classes, permut)
  print_table_footer()

def print_table_header(caption, label):
  print '\\begin{table}[!htb]'
  print '\\footnotesize'
  print '\\centering'
  print '\\caption{{{0}}}'.format(caption)
  print '\\label{{{0}}}'.format(label)
  print '\\begin{tabular}{r|c|c|c|c|c}'
  print '\\hline\\hline'
  print 'Classifier & Acc (\\%) & SC (\\%) & BH (\\%) & F-medida & MCC \\\\ \\hline'

def print_table_footer():
  print '\\hline\\hline'
  print '\\end{tabular}'
  print '\\end{table}'


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print 'Usage: {0} filename'.format(sys.argv[0])
  else:
    experiment(sys.argv[1])
