import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class BaseClassification(object):
  """ Base class for classification """

  def __init__(self, filename):

    content_list = []
    label_list = []

    # Reading and parsing CSV file
    with open(filename, 'rb') as csvfile:
      reader = csv.reader(csvfile)
      reader.next() # Skiping the header

      for row in reader:
        content_list.append(row[3])
        label_list.append(int(row[4]))

    content_list = np.asarray(content_list)
    label_list = np.asarray(label_list)

    # To maintain the original ratio between the training and test sets
    self.spam_list = content_list[label_list == 1]
    self.ham_list = content_list[label_list == 0]

    self.ones_list = label_list[label_list == 1]
    self.zeros_list = label_list[label_list == 0]


class SingleClassification(BaseClassification):
  """ Regular supervised training
      1- train with the first oldest labeled comments (below train_percent)
      2- test with the remaning comments (above train_percent)
      The dataset must be ordered by date (first = oldest)
  """

  def __init__(self, filename, clf, train_percent=0.8):
    super(SingleClassification, self).__init__(filename)
    self.clf = clf

    train_index_spam = int(len(self.spam_list) * train_percent)
    train_index_ham = int(len(self.ham_list) * train_percent)

    self.X_train = np.concatenate([self.spam_list[ :train_index_spam], self.ham_list[ :train_index_ham]])
    self.X_test = np.concatenate([self.spam_list[train_index_spam: ], self.ham_list[train_index_ham: ]])

    self.y_train = np.concatenate([self.ones_list[ :train_index_spam], self.zeros_list[ :train_index_ham]])
    self.y_test = np.concatenate([self.ones_list[train_index_spam: ], self.zeros_list[train_index_ham: ]])

  def classify(self):
    # Preparing bag of words
    vectorizer = CountVectorizer(min_df=1)
    bow_X = vectorizer.fit_transform(self.X_train)

    # Evaluating supervised training
    self.clf.fit(bow_X, self.y_train)
    bow_X = vectorizer.transform(self.X_test)
    y_pred = self.clf.predict(bow_X)

    return self.y_test, y_pred


class DualClassification(BaseClassification):
  """ Dual supervised training
      1- train with the first oldest labeled comments (below train_percent)
      2- label more comments with the intermediate classifier (between train_percent and ss_percent)
      3- test final classifier with the remaning comments (above ss_percent)
      The dataset must be ordered by date (first = oldest)
  """

  def __init__(self, filename, interm_clf, final_clf, threshold=None, train_percent=0.8, ss_percent=0.1):
    super(DualClassification, self).__init__(filename)
    self.interm_clf = interm_clf
    self.final_clf = final_clf
    self.threshold = threshold

    train_index_spam = int(len(self.spam_list) * train_percent)
    train_index_ham = int(len(self.ham_list) * train_percent)
    ss_index_spam = int(len(self.spam_list) * (train_percent+ss_percent))
    ss_index_ham = int(len(self.ham_list) * (train_percent+ss_percent))

    self.X_train = np.concatenate([self.spam_list[ :train_index_spam], self.ham_list[ :train_index_ham]])
    self.X_ss = np.concatenate([self.spam_list[train_index_spam:ss_index_spam],
                                self.ham_list[train_index_ham:ss_index_ham]])
    self.X_test = np.concatenate([self.spam_list[ss_index_spam: ], self.ham_list[ss_index_ham: ]])

    self.y_train = np.concatenate([self.ones_list[ :train_index_spam], self.zeros_list[ :train_index_ham]])
    self.y_ss = np.concatenate([self.ones_list[train_index_spam:ss_index_spam],
                                self.zeros_list[train_index_ham:ss_index_ham]])
    self.y_test = np.concatenate([self.ones_list[ss_index_spam: ], self.zeros_list[ss_index_ham: ]])

  def classify(self):

    # ======================== Intermediate classifier =========================
    X, y = self.prepare_intermediate()

    # Preparing bag of words
    vectorizer = CountVectorizer(min_df=1)
    bow_X = vectorizer.fit_transform(X)

    self.interm_clf.fit(bow_X, y)
    bow_X = vectorizer.transform(self.X_ss)
    self.y_pred_interm = self.interm_clf.predict(bow_X)

    if self.threshold:
      y_proba_interm = self.interm_clf.predict_proba(bow_X)
      above_threshold = [each[self.y_pred_interm[i]] >= self.threshold for i, each in enumerate(y_proba_interm)]
      self.above_threshold = np.asarray(above_threshold)

    # ============================ Final classifier ============================
    X, y = self.prepare_final()

    # Preparing bag of words
    vectorizer = CountVectorizer(min_df=1)
    bow_X = vectorizer.fit_transform(X)

    self.final_clf.fit(bow_X, y)
    bow_X = vectorizer.transform(self.final_test())
    self.y_pred_final = self.final_clf.predict(bow_X)

    return self.return_prediction() # Results of both trainings

  def prepare_intermediate(self):
    return self.X_train, self.y_train

  def prepare_final(self):
    if self.threshold:
      X = np.concatenate([self.X_train, self.X_ss[self.above_threshold]])
      y = np.concatenate([self.y_train, self.y_pred_interm[self.above_threshold]])
    else:
      X = np.concatenate([self.X_train, self.X_ss])
      y = np.concatenate([self.y_train, self.y_pred_interm])

    return X, y

  def final_test(self):
    if self.threshold:
      return np.concatenate([self.X_ss[~self.above_threshold], self.X_test])
    else:
      return self.X_test

  def return_prediction(self):
    # The scores should consider all errors

    if self.threshold:
      y_true = np.concatenate([self.y_ss[self.above_threshold],
                               self.y_ss[~self.above_threshold],
                               self.y_test])
      y_pred = np.concatenate([self.y_pred_interm[self.above_threshold], self.y_pred_final])

    else:
      y_true = np.concatenate([self.y_ss, self.y_test])
      y_pred = np.concatenate([self.y_pred_interm, self.y_pred_final])

    return y_true, y_pred

class SemiSupervisedClassification(DualClassification):
  """ Modify the DualClassification to use the LabelPropagation and LabelSpreading
  classifiers. Both algorithms require unlabeled data (-1) while fitting the data
  """

  def prepare_intermediate(self):
    # Semi-supervised labels (-1)
    unlabeled_data = np.repeat(-1, len(self.X_ss))

    X = np.concatenate([self.X_train, self.X_ss])
    y = np.concatenate([self.y_train, unlabeled_data])

    return X, y
