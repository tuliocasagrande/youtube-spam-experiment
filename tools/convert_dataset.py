import unicodecsv as csv
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer


def convert(fileprefix):

    content_list = []
    label_list = []

    with open(os.path.join('data_csv', fileprefix+'.csv'), 'rb') as csvfile:
        reader = csv.reader(csvfile)
        reader.next()  # Skipping the header

        for row in reader:
            content_list.append(row[3])
            label_list.append(int(row[4]))

    X = np.asarray(content_list)
    y = np.asarray(label_list)

    vectorizer = CountVectorizer()
    bow = vectorizer.fit_transform(X)

    header = vectorizer.get_feature_names() + ['CLASS']
    bow_array = bow.toarray()

    assert len(bow_array) == len(y)

    with open(os.path.join('data_bow', fileprefix+'-bow.csv'), 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for i, row in enumerate(bow_array):
            writer.writerow(np.concatenate((row, [y[i]])))

    print 'Dataset:', fileprefix
    print '# tokens:', len(header)-1
    print '# samples:', len(y)


if __name__ == '__main__':

    if not os.path.exists('data_bow'):
        os.makedirs('data_bow')

    file_list = ['example',
                 '01-9bZkp7q19f0',
                 '04-CevxZvSJLk8',
                 '07-KQ6zr6kCPj8',
                 '08-uelHwf8o7_U',
                 '09-pRpeEdMmmQ0']

    for fileprefix in file_list:
        convert(fileprefix)
