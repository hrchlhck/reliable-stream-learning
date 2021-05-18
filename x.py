from constants import MONTHS
from csv import DictReader
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python x.py CSV DIR")
        exit(1)
    
    filename = Path(sys.argv[1])
    output_dir = Path('images/')

    if not output_dir.exists():
        output_dir.mkdir()

    for file in filename.glob('*.csv'):
        if not 'old' in file.name:
            df = pd.read_csv(file)

            year_range = list(map(lambda x: int(x), file.name.split('_')[0:2]))

            df['test_month_year'] = np.nan
            df = df.assign(test_month_year = df.month.astype(str) + "/" + df.test_year.astype(str))
            df = df.drop_duplicates()

            df_1 = df[df['test_year'] == year_range[0]].copy()
            df_2 = df[df['test_year'] == year_range[1] - 1].copy()

            # Switches Jan at last row to first row
            jan_col = df_2[df_2['month'] == 'Jan']
            df_2.drop(index=jan_col.index, inplace=True)
            df_2 = pd.concat([jan_col, df_2], axis=0)

            # Merges year_range[0] with year_range[1]
            df = pd.concat([df_1, df_2], axis=0)
            df.shift(periods=1)

            fig = plt.figure()
            gs = fig.add_gridspec(2, 1, hspace=0.5)
            (ax1, ax2) = gs.subplots(sharex='row', sharey='col')

            ax1.plot(df_1['month'], df_1['fpr'], marker='X', label='FPR', color='lightseagreen')
            ax1.plot(df_1['month'], df_1['fnr'], marker='s', label='FNR', color='coral')
            ax1.grid(True)
            ax1.legend()
            ax1.set_title(f'FN Rate vs FP Rate over time in {year_range[0]}')

            ax2.plot(df_2['month'], df_2['fpr'], marker='X', label='FPR', color='lightseagreen')
            ax2.plot(df_2['month'], df_2['fnr'], marker='s', label='FNR', color='coral')
            ax2.grid(True)
            ax2.legend()
            ax2.set_title(f'FN Rate vs FP Rate over time in {year_range[1] - 1}')

            plt.savefig(f"images/{file.name[:-4]}.png", dpi=210)

            plt.show()
