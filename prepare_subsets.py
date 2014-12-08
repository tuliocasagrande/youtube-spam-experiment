#!/usr/bin/python
# This Python file uses the following encoding: utf-8

import sys
from os.path import splitext
import numpy as np

# This is a auxiliar script to create subsets. The original dataset has 150 spam
# and 150 ham. I want to create 5 subsets, with 200 samples each:
## 1- 50 spam and 150 ham
## 2- 75 spam and 125 ham
## 3- 100 spam and 100 ham
## 4- 125 spam and 75 ham
## 5- 150 spam and 50 ham

def parse(path):

  # Reading original file
  with open(path) as f:

    filepath = splitext(path)[0]
    f_50 = open(filepath + '-050.csv', 'w')
    f_75 = open(filepath + '-075.csv', 'w')
    f_100 = open(filepath + '-100.csv', 'w')
    f_125 = open(filepath + '-125.csv', 'w')
    f_150 = open(filepath + '-150.csv', 'w')

    # The experiment will perform a 80%/20% holdout. It needs some arrangements
    # so that the testing set has the same proportion of spam/ham as the training set:
    ##                  SPAM (%)  HAM (%)
    ## much more ham    25        75
    ## more ham         37.5      62.5
    ## balanced         50        50
    ## more spam        62.5      37.5
    ## much more spam   75        25

    index = range(150)

    index50 = np.concatenate([index[:40], index[-20:-15], index[-5:]])
    index75 = np.concatenate([index[:60], index[-20:-12], index[-7:]])
    index100 = np.concatenate([index[:80], index[-20:]])
    index125 = np.concatenate([index[:100], index[-25:]])
    index150 = index

    # Writing the header
    header = f.readline()
    f_50.write(header)
    f_75.write(header)
    f_100.write(header)
    f_125.write(header)
    f_150.write(header)

    ham = 0
    spam = 0

    for line in f:

      if line.rstrip()[-1] == '1':
        write_csv( f_50, line, spam, index50)
        write_csv( f_75, line, spam, index75)
        write_csv(f_100, line, spam, index100)
        write_csv(f_125, line, spam, index125)
        write_csv(f_150, line, spam, index150)
        spam += 1
      else:
        write_csv( f_50, line, ham, index150)
        write_csv( f_75, line, ham, index125)
        write_csv(f_100, line, ham, index100)
        write_csv(f_125, line, ham, index75)
        write_csv(f_150, line, ham, index50)
        ham += 1

    print spam
    print ham

  f_50.close()
  f_75.close()
  f_100.close()
  f_125.close()
  f_150.close()

def write_csv(f, line, i, index):
  if i in index:
    f.write(line)

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print 'Usage: {0} filename'.format(sys.argv[0])
  else:
    parse(sys.argv[1])
