from utils import show_results
from constants import CSV_PATH, IMAGES_PATH, MONTHS
from log import get_logger

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib import ticker

import seaborn as sns

from pathlib import Path

import pandas as pd
import re

__all__ = ['plot_results']

split_upper = lambda s: list(filter(None, re.split("([A-Z][^A-Z]*)", s)))

LOGGER = get_logger('plot')

def load_custom_fonts():
    fonts_path = ['/usr/local/share/fonts/p/']
    fonts = mpl.font_manager.findSystemFonts(fontpaths=fonts_path, fontext='ttf')

    for font in fonts:
        font_manager.fontManager.addfont(font)

def plot_results(filename: Path):
    filename = Path(filename)

    images_path = IMAGES_PATH.joinpath('final')

    # Setting up output directory
    if not images_path.exists():
        images_path.mkdir()

    df = pd.read_csv(filename)

    months = df['month']

    plt.figure(figsize=(6, 6))
    plt.plot(months, df['fpr'] * 100, marker='^', label='FP', ms=13, linestyle='dotted', color='red')
    plt.plot(months, df['fnr'] * 100, marker='s', label='FN', ms=13, linestyle='dotted', color='black')
    if 'rejection_rate' in df.columns:
        plt.plot(months, df['rejection_rate'] * 100, label='Rejection Rate', alpha=0.6, color='g')
        plt.fill_between(months, (df['rejection_rate']) * 100, alpha=0.2, color='g')
        plt.plot(months, (1 - df['rejection_rate']) * 100, label='Acceptance Rate', alpha=0.6, color='blue')
    plt.ylim((0, 100))
    plt.yticks([i * 10 for i in range(0, 11)])
    plt.xticks(rotation=60)

    # Position of X and Y labels
    ax = plt.gca()
    ax.yaxis.set_label_coords(-0.09, 0.5)
    ax.xaxis.set_label_coords(0.5 , -0.15)

    # Add more spacing between X ticks
    ax.set_xticks([i - 0.5 for i in range(12)]) 
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.xlabel("Month")
    plt.ylabel("Rate %")
    plt.legend(loc=2)
    # plt.title(f'FN Rate vs FP Rate over {year_range[0]} for {clf_name} in view {view}')

    # ax2.plot(df_2['month'], df_2['fpr'], marker='X', label='FPR', color='lightseagreen')
    # ax2.plot(df_2['month'], df_2['fnr'], marker='s', label='FNR', color='coral')
    # ax2.grid(True)
    # ax2.legend()
    # ax2.set_title(f'FN Rate vs FP Rate over time in {year_range[1]}')

    plt.savefig(images_path.joinpath(filename.name[:-4] + ".png"), dpi=210, bbox_inches='tight')

    LOGGER.info(f"Created plot {filename.name[:-4]}")

def plot_pareto():
    # Color cycle
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["mediumblue", "#de8f05", "#029e73"]) 

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
        plt.legend(loc=1)
    plt.yticks([i * 5 for i in range(0, 5)])
    plt.xticks([i * 5 for i in range(0, 5)])
    plt.xlabel('Reject rate %')
    plt.ylabel('Error rate %')
    plt.savefig(output_path.joinpath("rejection_curve.png"), dpi=210, bbox_inches='tight')

if __name__ == '__main__':
    load_custom_fonts()
    font = {
        'family': 'Palatino',
        'size': 22,
    }
    plt.rc('font', **font)
    
    for f in CSV_PATH.joinpath('classify_by_rejection').glob("*.csv"):
        if not f.name.startswith('old'):
            plot_results(f)