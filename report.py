# This Python file uses the following encoding: utf-8

import matplotlib
matplotlib.use('Agg')

import csv
import matplotlib.pyplot as plt
import numpy as np


def tex_report(filename, video_title, scores_list):
    caption = 'Resultados dos métodos de aprendizado de máquina para o ' \
              'vídeo{0}.'.format(video_title)
    label = 'tab:{0}'.format(video_title)

    s = '\\begin{table}[!htb]\n'
    s += '\\centering\n'
    s += '\\caption{{{0}}}\n'.format(caption)
    s += '\\label{{{0}}}\n'.format(label)
    s += '\\begin{tabular}{r|c|c|c|c|c|c|c|c|c|c}\n'
    s += '\\hline\\hline\n'
    s += 'Classifier & Acc (\\%) & SC (\\%) & BH (\\%) & '
    s += 'F-medida & MCC & Kappa '
    s += '& TP & TN & FP & FN '
    s += '\\\\ \\hline\n'

    for clf_title, sc in scores_list:
        s += '{0} & '.format(clf_title.replace('&', '\&'))
        s += '{0:.2f} & '.format(sc['acc'] * 100)
        s += '{0:.2f} & '.format(sc['sc'] * 100)
        s += '{0:.2f} & '.format(sc['bh'] * 100)
        s += '{0:.3f} & '.format(sc['f1'])
        s += '{0:.3f} & '.format(sc['mcc'])
        s += '{0:.3f} & '.format(sc['kappa'])
        s += '{0} & {1} & '.format(sc['tp'], sc['tn'])
        s += '{0} & {1} '.format(sc['fp'], sc['fn'])
        s += '\\\\ \n'

    s += '\\hline\\hline\n\\end{tabular}\n\\end{table}\n'

    with open(filename, 'w') as output_file:
        output_file.write(s)


def plot_bars(figurename, video_title, scores_list, metric):
    plt.figure()
    plt.title(video_title)
    plt.xlabel(metric.upper())

    performance = [scores[metric] for clf_title, scores in scores_list]
    classifiers = tuple(clf_title for clf_title, scores in scores_list)
    y_pos = np.arange(len(classifiers))
    plt.yticks(y_pos, classifiers)

    bar = plt.barh(y_pos, performance, align='center', alpha=0.4)
    best = performance[0]
    for i, p in enumerate(performance):
        if p == best:
            bar[i].set_color('r')

    plt.xticks(np.arange(0, 1.1, 0.1))  # guarantee an interval [0,1]
    plt.savefig(figurename + '_mcc.png', bbox_inches='tight')
    plt.savefig(figurename + '_mcc.pdf', bbox_inches='tight')
    plt.close()


def plot_roc(figurename, video_title, scores_list):
    plt.figure()
    plt.title(video_title)
    for clf_title, scores in scores_list:
        plt.plot(
            scores['fpr'], scores['tpr'],
            label=clf_title+' (1 - AUC = %0.2f)' % scores['roc_oneless_auc'])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    plt.savefig(figurename + '_roc.png', bbox_inches='tight')
    plt.savefig(figurename + '_roc.pdf', bbox_inches='tight')
    plt.close()


class CsvReport:

    def __init__(self, filename, clf_list, metric):
        self.filename = filename
        self.clf_list = clf_list
        self.metric = metric
        with open(filename, 'w') as f:
            csv.writer(f).writerow(['Video'] + clf_list)

    def report(self, video_title, scores_list):
        precision = 10

        scores_dict = {}
        for clf_title, sc in scores_list:
            scores_dict[clf_title] = sc

        row = [video_title]
        for clf in self.clf_list:
            row.append(round(scores_dict[clf][self.metric], precision))

        with open(self.filename, 'a') as f:
            csv.writer(f).writerow(row)
