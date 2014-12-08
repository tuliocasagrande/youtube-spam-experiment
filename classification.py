import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class BaseClassification(object):
  """ Base class for classification """

  def __init__(self, filename):

    contents = []
    classes = []

    # Reading and parsing CSV file
    with open(filename, 'rb') as csvfile:
      reader = csv.reader(csvfile)
      reader.next() # Skiping the header

      for row in reader:
        contents.append(row[3])
        classes.append(int(row[4]))

    self.contents = np.asarray(contents)
    self.classes = np.asarray(classes)


class SingleClassification(BaseClassification):
  """ Regular supervised training
      1- train with the 80%  oldest labeled comments (160)
      2- test with the remaning 20%  comments (40)

      The dataset must be ordered by date (first = oldest)
  """

  def __init__(self, filename, clf):
    super(SingleClassification, self).__init__(filename)
    self.clf = clf

    train_index = int(len(self.classes) * 0.8)

    self.X_train = self.contents[:train_index]
    self.y_train = self.classes[:train_index]

    self.X_test = self.contents[train_index:]
    self.y_test = self.classes[train_index:]

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
      1- train with the 80%  oldest labeled comments (160)
      2- predict half the remaining comments with the intermediate classifier (20)
      3- test final classifier with the other half (20)
  """

  def __init__(self, filename, interm_clf, final_clf, threshold=None):
    super(DualClassification, self).__init__(filename)
    self.interm_clf = interm_clf
    self.final_clf = final_clf
    self.threshold = threshold

    # Just to keep it dynamically, 160 = 80% of 200 total samples
    train_index = int(len(self.classes) * 0.8)
    semisupervised_index = int(len(self.classes) * 0.9)

    # The dataset is ordered by date (first = oldest)
    self.X_train = self.contents[:train_index]
    self.y_train = self.classes[:train_index]

    self.X_semisupervised = self.contents[train_index:semisupervised_index]
    self.y_semisupervised = self.classes[train_index:semisupervised_index]

    self.X_test = self.contents[semisupervised_index:]
    self.y_test = self.classes[semisupervised_index:]

  def classify(self):

    # ======================== Intermediate classifier =========================
    X, y = self.prepare_intermediate()

    # Preparing bag of words
    vectorizer = CountVectorizer(min_df=1)
    bow_X = vectorizer.fit_transform(X)

    self.interm_clf.fit(bow_X, y)
    bow_X = vectorizer.transform(self.X_semisupervised)
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
      X = np.concatenate([self.X_train, self.X_semisupervised[self.above_threshold]])
      y = np.concatenate([self.y_train, self.y_pred_interm[self.above_threshold]])
    else:
      X = np.concatenate([self.X_train, self.X_semisupervised])
      y = np.concatenate([self.y_train, self.y_pred_interm])

    return X, y

  def final_test(self):
    if self.threshold:
      return np.concatenate([self.X_semisupervised[~self.above_threshold], self.X_test])
    else:
      return self.X_test

  def return_prediction(self):
    # The scores should consider all errors

    if self.threshold:
      y_true = np.concatenate([self.y_semisupervised[self.above_threshold],
                               self.y_semisupervised[~self.above_threshold],
                               self.y_test])
      y_pred = np.concatenate([self.y_pred_interm[self.above_threshold], self.y_pred_final])

    else:
      y_true = np.concatenate([self.y_semisupervised, self.y_test])
      y_pred = np.concatenate([self.y_pred_interm, self.y_pred_final])

    return y_true, y_pred

class SemiSupervisedClassification(DualClassification):
  """ Modify the DualClassification to use the LabelPropagation and LabelSpreading
  classifiers. Both algorithms require unlabeled data (-1) while fitting the data
  """

  def prepare_intermediate(self):
    # Semi-supervised labels (-1)
    unlabeled_data = np.repeat(-1, len(self.X_semisupervised))

    X = np.concatenate([self.X_train, self.X_semisupervised])
    y = np.concatenate([self.y_train, unlabeled_data])

    return X, y
