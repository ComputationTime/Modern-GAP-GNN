import re
import getopt, sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_results(filepath):
    file = open(filepath)
    lines = file.readlines()
    table = {'agg_func': [], 'dataset': [], 'eps': [], 'encoder_acc': [], 'model_acc': []}
    result = None

    for idx, line in enumerate(lines):
        if not result:
            args = line.split()
            result = {'agg_func': args[0], 'dataset': args[1], 'eps': float(args[2]), 'encoder_acc': None, 'model_acc': None}
            continue
        accuracy = re.search('Avg Accuracy: ((\d|\.)+)', line)
        if accuracy:
            if not result['encoder_acc']:
                result['encoder_acc'] = float(accuracy.group(1))
            else:
                result['model_acc'] = float(accuracy.group(1))
        if result['model_acc']:
            table['agg_func'].append(result['agg_func'])
            table['dataset'].append(result['dataset'])
            table['eps'].append(result['eps'])
            table['encoder_acc'].append(result['encoder_acc'])
            table['model_acc'].append(result['model_acc'])
            result = None

    df = pd.DataFrame.from_dict(table)
    datasets = df['dataset'].unique()
    agg_funcs = df['agg_func'].unique()
    f = plt.figure(figsize=(len(datasets)*4,4))
    for i in range(len(datasets)):
        dataset = datasets[i]
        plt.subplot(1, len(datasets), i+1)
        for j in range(len(agg_funcs)):
            agg_func = agg_funcs[j]
            data = df.loc[(df['dataset'] == dataset) & (df['agg_func'] == agg_func)]
            x = data['eps']
            y = data['model_acc']
            plt.scatter(x, y, label=agg_func.upper())
            # plt.plot(x, y)
        plt.title(dataset.capitalize())
        plt.xlabel('Epsilon')
        plt.legend(loc='lower right')
        if i == 0:
            plt.ylabel('Model accuracy')
    plt.tight_layout()
    plt.savefig('plot.png')

    # if chosen_dataset != None:
    #     df = df[df['dataset'] == chosen_dataset]
    #     df = pd.pivot_table(df, values='throughput', index=['system'], columns=['grow'], aggfunc=np.mean)
    #
    #     for index, row in df.iterrows():
    #         plt.scatter(row.index,row.values,label=index)
    #         plt.plot(row.index, row.values)
    #     plt.legend()
    #     plt.yscale('log')
    #     plt.title(title + " on " + chosen_dataset)
    #     plt.xlabel('Grow coefficient')
    #     plt.ylabel('Throughput (edges/s)')
    #     plt.savefig('plot.png')

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
