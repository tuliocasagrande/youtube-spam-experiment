# This Python file uses the following encoding: utf-8

import numpy as np
import os

class SingleClassification:

    def __init__(self, src_folder, video_title, vectorizer):

        pos_train = open(os.path.join(src_folder, video_title + '-pos-train.txt')).readlines()
        neg_train = open(os.path.join(src_folder, video_title + '-neg-train.txt')).readlines()
        pos_test = open(os.path.join(src_folder, video_title + '-pos-test.txt')).readlines()
        neg_test = open(os.path.join(src_folder, video_title + '-neg-test.txt')).readlines()

        pos_train = np.asarray(pos_train)
        neg_train = np.asarray(neg_train)
        pos_test = np.asarray(pos_test)
        neg_test = np.asarray(neg_test)

        y_pos_train = np.repeat(1, len(pos_train))
        y_neg_train = np.repeat(0, len(neg_train))
        y_pos_test = np.repeat(1, len(pos_test))
        y_neg_test = np.repeat(0, len(neg_test))

        self.X_train = np.concatenate([pos_train, neg_train])
        self.X_test = np.concatenate([pos_test, neg_test])
        self.y_train = np.concatenate([y_pos_train, y_neg_train])
        self.y_test = np.concatenate([y_pos_test, y_neg_test])

        # Preparing bag of words
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

        return self.y_test, y_pred
