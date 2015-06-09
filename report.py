# This Python file uses the following encoding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import os

class Report(object):
  """docstring for Report"""
  def __init__(self, output_file, figures_path):
    self.output_file = output_file
    self.figures_path = figures_path


  def new_table(self, video_title):
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

    self.output_file.write(s)
    self.scores_list = []
    self.video_title = video_title


  def append_scores(self, clf_title, scores):
    self.scores_list.append((clf_title, scores))


  def print_scores(self):
    self.scores_list.sort(key=lambda scores: scores[1]['f1'], reverse=True)

    for clf_title, sc in self.scores_list:
      s = '{0} & '.format(clf_title.replace('&','\&'))
      s += '{0:.2f} & {1:.2f} & {2:.2f} & '.format(sc['acc'] * 100, sc['sc'] * 100, sc['bh'] * 100)
      # s += '{0:.3f} & {1:.3f} & {2:.3f} & '.format(sc['f1'], sc['mcc'], sc['kap'])
      s += '{0:.3f} & {1:.3f} & '.format(sc['f1'], sc['mcc'])
      s += '{0} & {1} & {2} & {3} \\\\ \n'.format(sc['tp'], sc['tn'], sc['fp'], sc['fn'])

      self.output_file.write(s)


  def print_table_footer(self):
    self.output_file.write('\\hline\\hline\n\\end{tabular}\n\\end{table}\n')


  def plot_figure(self, figure_name=None):
    if not figure_name:
      figure_name = self.video_title

    plt.figure()
    plt.title(figure_name)
    plt.xlabel('F-medida')

    performance = [scores['f1'] for clf_title, scores in self.scores_list]
    classifiers = tuple(clf_title for clf_title, scores in self.scores_list)
    y_pos = np.arange(len(classifiers))
    plt.yticks(y_pos, classifiers)

    bar = plt.barh(y_pos, performance, align='center', alpha=0.4)
    best = performance[0]
    for i, p in enumerate(performance):
      if p == best:
        bar[i].set_color('r')

    plt.xticks(np.arange(0, 1.1, 0.1)) # guarantee an interval [0,1]
    plt.savefig(os.path.join(self.figures_path, figure_name + '.png'), bbox_inches='tight')
    plt.savefig(os.path.join(self.figures_path, figure_name + '.pdf'), bbox_inches='tight')
