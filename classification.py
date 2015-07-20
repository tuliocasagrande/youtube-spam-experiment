import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics  import accuracy_score, auc, f1_score, matthews_corrcoef, roc_curve
# from skll.metrics import kappa # Skll 1.0.1 (current version) doesnt support scikit-learn 0.16

class BaseClassification(object):
  """ Base class for classification """

  def __init__(self, filename, stratified):

    content_list = []
    label_list = []

    # Reading and parsing CSV file
    with open(filename, 'rb') as csvfile:
      reader = csv.reader(csvfile)
      reader.next() # Skiping the header

      for row in reader:
        content_list.append(row[3])
        label_list.append(int(row[4]))

    self.X = np.asarray(content_list)
    self.y = np.asarray(label_list)

    # Ensure that the lists are both the same length
    assert(len(self.X) == len(self.y))

    if stratified:
      # Maintain the original ratio between the training and test sets

      self.X_spam = self.X[self.y == 1]
      self.y_spam = self.y[self.y == 1]

      self.X_ham = self.X[self.y == 0]
      self.y_ham = self.y[self.y == 0]



class SingleClassification(BaseClassification):
  """ Regular supervised training
      1- train with the first oldest labeled comments (below train_percent)
      2- test with the remaning comments (above train_percent)
      The dataset must be ordered by date (first = oldest)
  """

  def __init__(self, filename, train_percent=0.7, test_percent=None, stratified=False):
    super(SingleClassification, self).__init__(filename, stratified)

    if not test_percent: test_percent = 1 - train_percent

    if stratified:
      train_index_spam = int(len(self.X_spam) * train_percent)
      train_index_ham  = int(len(self.X_ham ) * train_percent)

      test_index_spam = int(len(self.X_spam) * (1-test_percent))
      test_index_ham  = int(len(self.X_ham ) * (1-test_percent))

      self.X_train = np.concatenate([self.X_spam[ :train_index_spam], self.X_ham[ :train_index_ham]])
      self.y_train = np.concatenate([self.y_spam[ :train_index_spam], self.y_ham[ :train_index_ham]])

      self.X_test = np.concatenate([self.X_spam[test_index_spam: ], self.X_ham[test_index_ham: ]])
      self.y_test = np.concatenate([self.y_spam[test_index_spam: ], self.y_ham[test_index_ham: ]])
    else:
      train_index = int(len(self.X) * train_percent)
      test_index = int(len(self.X) * (1-test_percent))

      self.X_train = self.X[ :train_index]
      self.y_train = self.y[ :train_index]
      self.X_test = self.X[test_index: ]
      self.y_test = self.y[test_index: ]

    # Preparing bag of words
    vectorizer = CountVectorizer(min_df=1)
    self.bow_train = vectorizer.fit_transform(self.X_train)
    self.bow_test = vectorizer.transform(self.X_test)

  def classify(self, clf):

    # Fitting and predicting
    try:
      clf.fit(self.bow_train, self.y_train)
      y_pred = clf.predict(self.bow_test)
    except TypeError:
      clf.fit(self.bow_train.toarray(), self.y_train)
      y_pred = clf.predict(self.bow_test.toarray())

    return self.y_test, y_pred, clf


class DualClassification(BaseClassification):
  """ Dual supervised training
      1- train with the first oldest labeled comments (below train_percent)
      2- label more comments with the intermediate classifier (between train_percent and ss_percent)
      3- test final classifier with the remaning comments (above ss_percent)
      The dataset must be ordered by date (first = oldest)
  """

  def __init__(self, filename, threshold=None, train_percent=0.3, ss_percent=0.4, stratified=False):
    super(DualClassification, self).__init__(filename, stratified)
    self.threshold = threshold

    if stratified:
      train_index_spam = int(len(self.X_spam) * train_percent)
      train_index_ham  = int(len(self.X_ham ) * train_percent)

      ss_index_spam = int(len(self.X_spam) * (train_percent+ss_percent))
      ss_index_ham  = int(len(self.X_ham ) * (train_percent+ss_percent))

      self.X_train = np.concatenate([self.X_spam[ :train_index_spam], self.X_ham[ :train_index_ham]])
      self.y_train = np.concatenate([self.y_spam[ :train_index_spam], self.y_ham[ :train_index_ham]])

      self.X_ss = np.concatenate([self.X_spam[train_index_spam:ss_index_spam], self.X_ham[train_index_ham:ss_index_ham]])
      self.y_ss = np.concatenate([self.y_spam[train_index_spam:ss_index_spam], self.y_ham[train_index_ham:ss_index_ham]])

      self.X_test = np.concatenate([self.X_spam[ss_index_spam: ], self.X_ham[ss_index_ham: ]])
      self.y_test = np.concatenate([self.y_spam[ss_index_spam: ], self.y_ham[ss_index_ham: ]])
    else:
      train_index = int(len(self.X) * train_percent)
      ss_index = int(len(self.X) * (train_percent+ss_percent))

      self.X_train = self.X[ :train_index]
      self.y_train = self.y[ :train_index]

      self.X_ss = self.X[train_index:ss_index]
      self.y_ss = self.y[train_index:ss_index]

      self.X_test = self.X[ss_index: ]
      self.y_test = self.y[ss_index: ]

  def classify(self, interm_clf, final_clf):

    # ======================== Intermediate classifier =========================
    X_train, y_train = self.prepare_intermediate()

    # Preparing bag of words
    vectorizer = CountVectorizer(min_df=1)
    bow_train = vectorizer.fit_transform(X_train)
    bow_ss = vectorizer.transform(self.X_ss)

    try:
      interm_clf.fit(bow_train, y_train)
      self.y_pred_interm = interm_clf.predict(bow_ss)
      y_proba_interm = interm_clf.predict_proba(bow_ss)
    except TypeError:
      interm_clf.fit(bow_train.toarray(), y_train)
      self.y_pred_interm = interm_clf.predict(bow_ss.toarray())
      y_proba_interm = interm_clf.predict_proba(bow_ss.toarray())

    if self.threshold:
      above_threshold = [proba[self.y_pred_interm[i]] >= self.threshold
                         for i, proba in enumerate(y_proba_interm)]
      self.above_threshold = np.asarray(above_threshold)

    # ============================ Final classifier ============================
    X_train, y_train = self.prepare_final()

    # Preparing bag of words
    vectorizer = CountVectorizer(min_df=1)
    bow_train = vectorizer.fit_transform(X_train)
    bow_test = vectorizer.transform(self.X_test)

    try:
      final_clf.fit(bow_train, y_train)
      self.y_pred_final = final_clf.predict(bow_test)
    except TypeError:
      final_clf.fit(bow_train.toarray(), y_train)
      self.y_pred_final = final_clf.predict(bow_test.toarray())

    # Result of the final training only
    return self.y_test, self.y_pred_final, interm_clf, final_clf, sum(above_threshold), len(self.X_ss)

  def prepare_intermediate(self):
    return self.X_train, self.y_train

  def prepare_final(self):
    if self.threshold:
      X_train = np.concatenate([self.X_train, self.X_ss[self.above_threshold]])
      y_train = np.concatenate([self.y_train, self.y_pred_interm[self.above_threshold]])
    else:
      X_train = np.concatenate([self.X_train, self.X_ss])
      y_train = np.concatenate([self.y_train, self.y_pred_interm])

    return X_train, y_train

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


def calculate_scores(y_true, y_pred):

  # Ensure that the lists are both the same length
  assert(len(y_true) == len(y_pred))

  scores = {}
  scores['tp'] = tp = sum((y_true == y_pred) & (y_true == 1))
  scores['tn'] = tn = sum((y_true == y_pred) & (y_true == 0))
  scores['fp'] = fp = sum((y_true != y_pred) & (y_true == 0))
  scores['fn'] = fn = sum((y_true != y_pred) & (y_true == 1))

  scores['acc'] = accuracy_score(y_true, y_pred)
  scores['f1'] = f1_score(y_true, y_pred)
  scores['mcc'] = matthews_corrcoef(y_true, y_pred)
  # scores['kap'] = kappa(y_true, y_pred)

  try:
    scores['sc'] = float(tp) / (tp + fn)
  except ZeroDivisionError:
    scores['sc'] = 0

  try:
    scores['bh'] = float(fp) / (fp + tn)
  except ZeroDivisionError:
    scores['bh'] = 0

  # Compute ROC curve and ROC area
  scores['fpr'], scores['tpr'], _ = roc_curve(y_true, y_pred)
  scores['roc_oneless_auc'] = 1 - auc(scores['fpr'], scores['tpr'])

  return scores
