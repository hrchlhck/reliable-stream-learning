import matplotlib
from constants import CSV_PATH, IMAGES_PATH, MONTHS
from log import get_logger
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import numpy as np
import re

__all__ = ['plot_results']

split_upper = lambda s: list(filter(None, re.split("([A-Z][^A-Z]*)", s)))

LOGGER = get_logger('plot')

def load_custom_fonts():
    fonts_path = ['/usr/local/share/fonts/p/']
    fonts = matplotlib.font_manager.findSystemFonts(fontpaths=fonts_path, fontext='ttf')

    for font in fonts:
        font_manager.fontManager.addfont(font)

def plot_results(filename: Path):
    filename = Path(filename)

    # Setting up output directory
    if not IMAGES_PATH.exists():
        IMAGES_PATH.mkdir()

    df = pd.read_csv(filename)

    # Parsing year and classifier name from file name
    year_range = list(map(lambda x: int(x), filename.name.split('_')[2:4]))
    clf_name = ' '.join(split_upper(filename.name.split('_')[1]))
    view = filename.name.split('_')[-1][:-4]

    # Creating new column as "month/year"
    df['test_month_year'] = np.nan
    df = df.assign(test_month_year = df.month.astype(str) + "/" + df.test_year.astype(str))
    df = df.drop_duplicates()

    # df_1 = df[df['test_year'] == year_range[0]].copy()
    # df_2 = df[df['test_year'] == year_range[1]].copy()

    # # Switches Jan at last row to first row
    # jan_col = df_2[df_2['month'] == 'Jan']
    # df_2.drop(index=jan_col.index, inplace=True)
    # df_2 = pd.concat([jan_col, df_2], axis=0)

    # # Merges year_range[0] with year_range[1]
    # df = pd.concat([df_1, df_2], axis=0)
    # df.shift(periods=1)

    months = df['month']
    font = {
        'family': 'Palatino',
        'size': 12,
    }
    plt.rc('font', **font)
    plt.figure(figsize=(5.5, 5.5))
    plt.plot(months, df['fpr'] * 100, marker='^', label='FPR', color='red', ms=10)
    plt.plot(months, df['fnr'] * 100, marker='s', label='FNR', color='black', ms=10)
    plt.ylim((0, 100))
    plt.yticks([i * 10 for i in range(0, 11)])
    plt.xlabel("Month")
    plt.ylabel("Rate %")
    plt.grid(True)
    plt.legend()
    # plt.title(f'FN Rate vs FP Rate over {year_range[0]} for {clf_name} in view {view}')

    # ax2.plot(df_2['month'], df_2['fpr'], marker='X', label='FPR', color='lightseagreen')
    # ax2.plot(df_2['month'], df_2['fnr'], marker='s', label='FNR', color='coral')
    # ax2.grid(True)
    # ax2.legend()
    # ax2.set_title(f'FN Rate vs FP Rate over time in {year_range[1]}')

    plt.savefig(f"images/{filename.name[:-4]}.png", dpi=210)

    LOGGER.info(f"Created plot {filename.name[:-4]}")


if __name__ == '__main__':
    load_custom_fonts()

    for file in CSV_PATH.glob('*.csv'):
        if '2014' in file.name:
            plot_results(file)