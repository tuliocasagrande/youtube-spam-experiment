import csv
import os
import random

with open(os.path.join('data_new', '08-Eminem-uelHwf8o7_U.csv'), 'rb') as fr:
  with open(os.path.join('data_new', '08-rotulada-tratada-embaralhada.csv'), 'wb') as fw:
    reader = csv.reader(fr)
    writer = csv.writer(fw)

    writer.writerow(reader.next()) # Writing header

    spam_list = []
    ham_list = []
    for row in reader:
      if row[4] == '1':
        spam_list.append(row)
      elif row[4] == '0':
        ham_list.append(row)
      else:
        raise Exception

    rand_list = [1] * len(spam_list) + [0] * len(ham_list)
    random.seed(0)
    random.shuffle(rand_list)

    for i in xrange(len(spam_list)):
      spam_list[i][2] = ''

    for r in rand_list:
      if r == 1:
        writer.writerow(spam_list.pop(0))
      elif r == 0:
        writer.writerow(ham_list.pop(0))


