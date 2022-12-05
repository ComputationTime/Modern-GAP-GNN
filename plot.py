import re
import getopt, sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_results(filepath):
    file = open(filepath)
    lines = file.readlines()
    table = {'agg_func': [], 'dataset': [], 'eps': [], 'encoder_acc': [], 'model_acc': [], 'att_acc': []}
    result = None

    for idx, line in enumerate(lines):

        test_line = re.search('Test Error:', line)
        epsilon_line = re.search('Epsilon:', line)
        accuracy = re.search('Avg Accuracy: ((\d|\.)+)', line)

        if not result or (not accuracy and not test_line and not epsilon_line):
            args = line.split()
            result = {'agg_func': args[0], 'dataset': args[1], 'eps': float(args[2]), 'encoder_acc': None, 'model_acc': None, 'att_acc': None}
            continue
        if accuracy:
            if not result['encoder_acc']:
                result['encoder_acc'] = float(accuracy.group(1))
            elif result['agg_func'] == 'pmat' and not result['att_acc']:
                result['att_acc'] = float(accuracy.group(1))
            else:
                result['model_acc'] = float(accuracy.group(1))
        if result['model_acc']:
            table['agg_func'].append(result['agg_func'])
            table['dataset'].append(result['dataset'])
            table['eps'].append(result['eps'])
            table['encoder_acc'].append(result['encoder_acc'])
            table['model_acc'].append(result['model_acc'])
            table['att_acc'].append(result['att_acc'])
            result = None

    df = pd.DataFrame.from_dict(table)
    y_min = df['model_acc'].min()-3
    y_max = df['model_acc'].max()+3
    datasets = df['dataset'].unique()
    agg_funcs = df['agg_func'].unique()
    f = plt.figure(figsize=(len(datasets)*4,4))
    for i in range(len(datasets)):
        dataset = datasets[i]
        plt.subplot(1, len(datasets), i+1)
        for j in range(len(agg_funcs)):
            agg_func = agg_funcs[j]
            data = df.loc[(df['dataset'] == dataset) & (df['agg_func'] == agg_func)]
            data_means = data[['eps', 'model_acc']].groupby('eps').agg('mean')
            x = data['eps']
            y = data['model_acc']
            y_mean = data_means['model_acc']
            plt.scatter(x, y, label=agg_func.upper())
            plt.plot(data_means.index, y_mean)
        plt.title(dataset.capitalize())
        plt.xlabel('Epsilon')
        plt.ylim(y_min, y_max)
        if i == 0:
            plt.ylabel('Model accuracy')
            plt.legend(loc='center right')
    plt.tight_layout()
    plt.savefig('plot.png')

if __name__ == "__main__":
  try:
      opts, args = getopt.getopt(sys.argv[1:], 'f:', ['filepath='])
  except getopt.GetoptError:
      print("Flag doesn't exist")
      sys.exit(2)

  filepath = None
  for o, a in opts:
      if o in ('-f', '--filepath'):
          filepath = a

  plot_results(filepath)
