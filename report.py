# This Python file uses the following encoding: utf-8

import csv
import matplotlib.pyplot as plt
import numpy as np
import os


def tex_report(filename, video_title, scores_list):
  caption = 'Resultados dos métodos de aprendizado de máquina para o vídeo {0}.'.format(video_title)
  label = 'tab:{0}'.format(video_title)

  s = '\\begin{table}[!htb]\n'
  s += '\\centering\n'
  s += '\\caption{{{0}}}\n'.format(caption)
  s += '\\label{{{0}}}\n'.format(label)
  s += '\\begin{tabular}{r|c|c|c|c|c|c|c|c|c|c}\n'
  s += '\\hline\\hline\n'
  # s += 'Classifier & Acc (\\%) & SC (\\%) & BH (\\%) & F-medida & MCC & Kappa & TP & TN & FP & FN \\\\ \\hline\n'
  s += 'Classifier & Acc (\\%) & SC (\\%) & BH (\\%) & F-medida & MCC & TP & TN & FP & FN \\\\ \\hline\n'

  for clf_title, sc in scores_list:
    s += '{0} & '.format(clf_title.replace('&','\&'))
    s += '{0:.2f} & {1:.2f} & {2:.2f} & '.format(sc['acc'] * 100, sc['sc'] * 100, sc['bh'] * 100)
    # s += '{0:.3f} & {1:.3f} & {2:.3f} & '.format(sc['f1'], sc['mcc'], sc['kap'])
    s += '{0:.3f} & {1:.3f} & '.format(sc['f1'], sc['mcc'])
    s += '{0} & {1} & {2} & {3} \\\\ \n'.format(sc['tp'], sc['tn'], sc['fp'], sc['fn'])

  s += '\\hline\\hline\n\\end{tabular}\n\\end{table}\n'

  with open(filename, 'w') as output_file:
    output_file.write(s)


def plot_mcc_bars(figurename, video_title, scores_list):
  plt.figure()
  plt.title(video_title)
  plt.xlabel('MCC')

  performance = [scores['mcc'] for clf_title, scores in scores_list]
  classifiers = tuple(clf_title for clf_title, scores in scores_list)
  y_pos = np.arange(len(classifiers))
  plt.yticks(y_pos, classifiers)

  bar = plt.barh(y_pos, performance, align='center', alpha=0.4)
  best = performance[0]
  for i, p in enumerate(performance):
    if p == best:
      bar[i].set_color('r')

  plt.xticks(np.arange(0, 1.1, 0.1)) # guarantee an interval [0,1]
  plt.savefig(figurename + '.png', bbox_inches='tight')
  plt.savefig(figurename + '.pdf', bbox_inches='tight')


class CsvReport:

  def __init__(self, filename, clf_list):
    self.filename = filename
    self.clf_list = clf_list
    with open(filename, 'w') as f:
      csv.writer(f).writerow(['Video'] + clf_list)

  def report(self, video_title, scores_list):
    scores_dict = {}
    for clf_title, sc in scores_list:
      scores_dict[clf_title] = sc

    row = [video_title]
    for clf in self.clf_list:
      row.append(scores_dict[clf]['mcc'])

    with open(self.filename, 'a') as f:
      csv.writer(f).writerow(row)
