from constants import CSV_PATH, IMAGES_PATH, MONTHS
from log import get_logger

from IPython import embed

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib import gridspec

from pathlib import Path

import pandas as pd
import numpy as np
import re

__all__ = ['plot_results']

split_upper = lambda s: list(filter(None, re.split("([A-Z][^A-Z]*)", s)))

LOGGER = get_logger('plot')

def load_custom_fonts():
    fonts_path = ['/usr/local/share/fonts/p/']
    fonts = mpl.font_manager.findSystemFonts(fontpaths=fonts_path, fontext='ttf')

    for font in fonts:
        font_manager.fontManager.addfont(font)

load_custom_fonts()
font = {
    'family': 'Palatino',
    'size': 22,
}
plt.rc('font', **font)

def plot_results(filename: Path):
    filename = Path(filename)

    images_path = IMAGES_PATH.joinpath(filename.parent.name)

    # Setting up output directory
    if not images_path.exists():
        images_path.mkdir()

    df = pd.read_csv(filename)

    months = [mn for m, mn in MONTHS]

    fig = plt.figure(figsize=(5, 5), constrained_layout=True)

    # fig.subplots_adjust(hspace=0.01)
    
    fpr_fnr = fig.add_subplot(111)
    gs = gridspec.GridSpec(3, 1)
    fpr_fnr.set_position(gs[0:4].get_position(fig))
    fpr_fnr.set_subplotspec(gs[0:4])

    fpr_fnr.plot(months, df['fpr'] * 100, marker='^', label='FP', ms=10, linestyle='dotted', color='red', alpha=0.8)
    fpr_fnr.plot(months, df['fnr'] * 100, marker='s', label='FN', ms=10, linestyle='dotted', color='black', alpha=0.8)
    fpr_fnr.set(ylim=(0, 80), yticks=[i * 10 for i in range(0, 9)], ylabel='Rate')
    fpr_fnr.set(xticks=months)
    fpr_fnr.tick_params(axis='x', rotation=60)
    fpr_fnr.set(xlabel="Month")
    # fpr_fnr.yaxis.set_label_coords(-0.14, 0)
    
    if 'rejection_rate' in df.columns:
        reject_plot = fig.add_subplot(gs[2])
        accept = (1 - df['rejection_rate']) * 100
        reject = df['rejection_rate'] * 100

        # 0, reject. 1, accept
        colors = ['#fe0014', '#55d77d']
        
        reject_plot.plot([], [], label='Reject', color=colors[0], linewidth=10)
        reject_plot.plot([], [], label='Accept', color=colors[1], linewidth=10)
        reject_plot.stackplot(months, reject, accept , colors=colors)

        reject_plot.set(ylim=(0, 5), yticks=[i * 25 for i in range(0, 5)])
        reject_plot.tick_params(axis='x', rotation=60)
        reject_plot.set(xlabel="Month")


    fig.legend(loc=2, prop={'size': 16}, bbox_to_anchor=(0.14, .86))
    fig.savefig(images_path.joinpath(filename.name[:-4] + ".svg"), dpi=290, bbox_inches='tight', format='svg')

    LOGGER.info(f"Created plot {filename.name[:-4]}")

def plot_pareto():
    # Color cycle
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["mediumblue", "red", "black"]) 

    output_path = IMAGES_PATH.joinpath('pareto_computed')

    if not output_path.exists():
        output_path.mkdir()

    # Load files
    files = [(split_upper(file.name[:-4]), pd.read_csv(file)) for file in CSV_PATH.joinpath("pareto_computed").glob("*.csv")]
    
    # Filtering columns
    files = [(pair[0], pair[1][['reject_rate', 'error_rate']]) for pair in files]

    plt.figure(figsize=(7, 7))
    for file in files:
        x = file[1]['reject_rate'] * 100
        y = file[1]['error_rate'] * 100

        plt.plot(x, y, label=' '.join(file[0][:-1]), linewidth=4, alpha=0.8)
        # plt.plot(x, yy, alpha=0.8, marker='x')
        plt.legend(loc=1)
    plt.yticks([i * 5 for i in range(0, 5)])
    plt.xticks([i * 5 for i in range(0, 5)])
    plt.xlim((0, 20))
    plt.xlabel('Reject rate %')
    plt.ylabel('Error rate %')
    plt.savefig(output_path.joinpath("rejection_curve.png"), dpi=210, bbox_inches='tight')

def plot_time(filename: Path):
    if isinstance(filename, str):
        filename = Path(filename)

    output_dir = IMAGES_PATH.joinpath(filename.parent.name)

    if not output_dir.exists():
        output_dir.mkdir()

    df = pd.read_csv(filename)

    # Getting month names
    months = [mn for _, mn in MONTHS]

    # Selecting train time elapsed per classifier
    train = df[df['type'] == 'train']

    # Selecting test type elapsed per classifier
    test = df[df['type'] == 'test'].copy()

    # Converting seconds to minutes at 'test'
    test['time_elapsed'] = test['time_elapsed'].map(lambda x: x / 60)

    rf = test[(test['clf'] == 'HoeffdingTreeClassifier')]
    gbt = test[(test['clf'] == 'LeveragingBaggingClassifier')]
    ada = test[(test['clf'] == 'OzaBaggingClassifier')]
    ens = test[(test['clf'] == 'StreamVotingClassifier')]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Plotting test time per month, per classifier
    conf = {'linewidth': 2, 'alpha': 0.8, 'marker': 'o', 'ms': 6}
    ax1.plot(months, rf['time_elapsed'], label='Random Forest', **conf)
    ax1.plot(months, gbt['time_elapsed'], label='Gradient Boosting', **conf)
    ax1.plot(months, ada['time_elapsed'], label='Ada Boost', **conf)

    # Plotting Ensmble with partial results
    if len(ens['time_elapsed'] < 12):
        y = ens['time_elapsed'].tolist()
        y += [10 for _ in range(12 - len(ens))]
        ax1.plot(months, y, label='Ensemble', **conf)
    else:
        ax1.plot(months, ens['time_elapsed'], label='Ensemble', **conf)

    ax1.set(xlim=(1, 11), xticks=months)
    ax1.set(ylim=(0, 240), yticks=[i * 30 for i in range(0, 9)])
    ax1.tick_params(axis='x', rotation=45)
    ax1.set(xlabel='Months', ylabel='Minutes')
    ax1.legend(loc=2, prop={'size': 14})

    # Plotting train time per classifier
    ax2.bar(['RF', 'GBT', 'ADA', 'Ens'], train['time_elapsed'], label='Time elapsed')
    ax2.set(ylim=(0, 80), yticks=[i * 15 for i in range(0, 7)])
    ax2.tick_params(axis='x', rotation=45)
    ax2.set(xlabel='Classifier', ylabel='Seconds')
    ax2.legend(loc=2, prop={'size': 14})

    plt.subplots_adjust(hspace=0.6)

    fig.savefig(output_dir.joinpath(filename.name[:-4] + ".png"), dpi=290, bbox_inches='tight')

if __name__ == '__main__':    
    for f in CSV_PATH.joinpath('stream_classifiers').glob("*.csv"):
        if not f.name.startswith('time'):
            plot_results(f)
    # plot_time('results/stream_classifiers/time_elapsed.csv')
