import matplotlib

from constants import CSV_PATH, IMAGES_PATH, MONTHS
from log import get_logger

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from pathlib import Path

import pandas as pd
import re

__all__ = ['fig2_3_4_5_10_11']

split_upper = lambda s: list(filter(None, re.split("([A-Z][^A-Z]*)", s)))

COMMON_PLOT_KWARGS = {'linestyle': 'dashed', 'ms': 10, 'alpha': 0.8}
COMMON_LEGEND_KWARGS = {'loc': 'upper center', 'prop': {'size': 16}, 'frameon': False}
COMMON_SAVEFIG_KWARGS = {'dpi': 290, 'bbox_inches': 'tight', 'transparent': True}

LOGGER = get_logger('plot')

def load_custom_fonts():
    fonts_path = ['/usr/local/share/fonts/p/']
    fonts = mpl.font_manager.findSystemFonts(fontpaths=fonts_path, fontext='ttf')

    for font in fonts:
        font_manager.fontManager.addfont(font)

load_custom_fonts()
font = {
    'family': 'Palatino',
    'size': 20,
}
plt.rc('font', **font)

TO_MILLION = matplotlib.ticker.FuncFormatter(lambda x, p: f'{int(x / 1_000_000)}')

def fig2_3_4_5_10_11(filename: Path):
    filename = Path(filename)

    images_path = IMAGES_PATH.joinpath(filename.parent.name)

    # Setting up output directory
    if not images_path.exists():
        images_path.mkdir()

    df = pd.read_csv(filename)

    months = list(MONTHS.values())

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)

    if 'rejection_rate' in df.columns:
        accept = (1 - df['rejection_rate']) * 100
        reject = df['rejection_rate'] * 100

        # 0, accept. 1, reject
        colors = ['#55d77d', '#fe0014']

        secax = ax.twinx()
        fpr = secax.plot(months, df['fpr'] * 120, marker='^', label='FP', color='red', **COMMON_PLOT_KWARGS, zorder=1)
        fnr = secax.plot(months, df['fnr'] * 120, marker='s', label='FN', color='black', **COMMON_PLOT_KWARGS)
        secax.set(xticks=months, xlim=(0, 11), xlabel="Month")
        secax.tick_params(axis='x', rotation=60)
        secax.set(ylabel='Error Rate (%)', ylim=(0, 80))

        secax.set_axisbelow(True)
        stacks = secax.stackplot(months, accept, reject, colors=colors, alpha=0.4, labels=['Accept', 'Reject'], zorder=-1)

        stacks[0].set_hatch('\\\\\\')
        stacks[1].set_hatch('///')

        stacks[0].set_edgecolor('white')
        stacks[1].set_edgecolor('white')
    
        secax.set(ylabel='Verifier Rate (%)', ylim=[0, 100])
        # lines += stacks

    # fpr = ax.plot(months, df['fpr'] * 100, marker='^', label='FP', color='red', **COMMON_PLOT_KWARGS, zorder=1)
    # fnr = ax.plot(months, df['fnr'] * 100, marker='s', label='FN', color='black', **COMMON_PLOT_KWARGS)
    ax.set(xticks=months, xlim=(0, 11), xlabel="Month")
    ax.tick_params(axis='x', rotation=60)

    ax.set(ylabel='Error Rate (%)', ylim=(0, 80))
    
    lines = fpr + fnr + stacks

    labels = [l.get_label() for l in lines]

    ax.legend(lines, labels, ncol=4, bbox_to_anchor=(0.5, 1.2), **COMMON_LEGEND_KWARGS)
    fig.savefig(images_path.joinpath(filename.name[:-4] + ".png"), **COMMON_SAVEFIG_KWARGS)
    fig.savefig(images_path.joinpath(filename.name[:-4] + ".svg"), **COMMON_SAVEFIG_KWARGS, format='svg')

    LOGGER.info(f"Created plot {filename.name[:-4]}")

def fig9():
    output_path = IMAGES_PATH.joinpath('fig9')

    if not output_path.exists():
        output_path.mkdir()

    # Load files
    files = [['', pd.read_csv(file)] for file in CSV_PATH.joinpath("pareto_computed").glob("*.csv")]
    
    COMMON_LEGEND_KWARGS['loc'] = 'upper right'

    # Renaming
    files[0][0] = 'Hoeffding Tree'
    files[1][0] = 'Leveraging Bagging'
    files[2][0] = 'Oza Bagging'

    # Filtering columns
    files = [(pair[0], pair[1][['reject_rate', 'error_rate']]) for pair in files]

    linestyles = ['dashdot', 'dotted', '-']
    colors = ['red', 'black', 'green']

    plt.figure(figsize=(7, 5), constrained_layout=True)

    for i, file in enumerate(files):
        df = file[1]
        df = df.sort_values(by='reject_rate')
        x = (df['reject_rate'] * 100).to_list()
        y = (df['error_rate'] * 100).to_list()

        x.append(100)
        y.append(0)
        x.insert(0, 0)
        y.insert(0, y[0])

        plt.plot(x, y, label=''.join(file[0]), color=colors[i], linestyle=linestyles[i], linewidth=3, alpha=0.8)

    plt.yticks([i * 5 for i in range(0, 5)])
    plt.xticks([i * 5 for i in range(0, 5)])
    plt.xlim((0, 20))
    plt.ylim((0, 15))
    plt.xlabel('Rejection Rate (%)')
    plt.ylabel('Average Error Rate (%)')
    plt.legend(**COMMON_LEGEND_KWARGS, ncol=1)
    plt.savefig(output_path.joinpath("rejection_curve.png"), **COMMON_SAVEFIG_KWARGS)
    plt.savefig(output_path.joinpath("rejection_curve.svg"), **COMMON_SAVEFIG_KWARGS, format='svg')

def fig1a(view: str, year: int):
    """ Plot density of instances per month on MAWILab dataset """

    output = IMAGES_PATH.joinpath('dataset_instances')

    if not output.exists():
        output.mkdir()

    df = pd.read_csv(CSV_PATH.joinpath(f'{view}_{year}_density.csv'))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)

    months = [month for month in MONTHS.values()]

    ax.plot([], [], label='Attack', color='red', linewidth=7)
    ax.plot([], [], label='Normal', color='gray', linewidth=7)
    stacks = ax.stackplot(months, df['attack'] * 100, df['normal'] * 100, colors=['red', 'gray'])

    # Setting texture
    stacks[0].set_hatch('\\\\\\')
    stacks[1].set_hatch('///')

    ax.set(xticks=months, xlim=(0, 11), xlabel='Month')
    ax.tick_params(axis='x', rotation=60)

    ax.set(ylabel='Network Flows (Million)', ylim=(0, 1_000_000_000))
    ax.get_yaxis().set_major_formatter(TO_MILLION)

    ax.legend(**COMMON_LEGEND_KWARGS, ncol=2, bbox_to_anchor=(0.5, 1.13))
    fig.savefig(output.joinpath(f'qtd_instancias.svg'), **COMMON_SAVEFIG_KWARGS, format='svg')
    fig.savefig(output.joinpath(f'qtd_instancias.png'), **COMMON_SAVEFIG_KWARGS)

def fig1b(view: str, year: int):
    output = IMAGES_PATH.joinpath('dataset_instances')

    if not output.exists():
        output.mkdir()

    df = pd.read_csv(CSV_PATH.joinpath(f'{view}_{year}_density.csv'))

    months = [month for month in MONTHS.values()]

    # 'Attack' and 'Normal' instances ratio per month
    for m in range(1, 13):
        df_temp = df[df['month'] == m]
        total = df_temp['attack'] + df_temp['normal']
        total = total.to_numpy()[0]
        normal_ratio = df_temp['normal'].to_numpy()[0] / total
        df.loc[m - 1, 'total_instances'] = total
        df.loc[m - 1, 'normal_ratio'] = normal_ratio
    
    df['attack_ratio'] = (1 - df['normal_ratio'])

    print(df['attack_ratio'])

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)

    ax.plot([], [], label='Attack', color='red', linewidth=7)
    ax.plot([], [], label='Normal', color='gray', linewidth=7)
    stacks = ax.stackplot(months, df['attack_ratio'] * 100, df['normal_ratio'] * 100, colors=['red', 'gray'])

    # Setting texture
    stacks[0].set_hatch('\\\\\\')
    stacks[1].set_hatch('///')

    ax.set(xticks=months, xlim=(0, 11), xlabel='Month')
    ax.tick_params(axis='x', rotation=60)

    ax.set(yticks=[i * 10 for i in range(0, 11)], ylim=(0, 100), ylabel='Network Flows (%)')

    ax.legend(**COMMON_LEGEND_KWARGS, ncol=2, bbox_to_anchor=(0.5, 1.13))
    fig.savefig(output.joinpath(f'perc_instancias.svg'), **COMMON_SAVEFIG_KWARGS, format='svg')
    fig.savefig(output.joinpath(f'perc_instancias.png'), **COMMON_SAVEFIG_KWARGS)

def fig6(view: str, year: int):
    output = IMAGES_PATH.joinpath('dataset_instances')

    if not output.exists():
        output.mkdir()

    df = pd.read_csv(CSV_PATH.joinpath(f'{view}_{year}_density.csv'))

    months = [month for month in MONTHS.values()]

    # Total instances per month
    for m in range(1, 13):
        df_temp = df[df['month'] == m]
        total = df_temp['attack'] + df_temp['normal']
        total = total.to_numpy()[0]
        df.loc[m - 1, 'total_instances'] = total

    # Cumulative dist
    total = df['total_instances']
    inst_per_month = dict()
    for m in range(1, 13):
        inst_per_month[m] = sum(total.loc[:m - 1])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)

    ax.bar(months, list(map(lambda x: x * 100, inst_per_month.values())), label='Number of Instances', color='blue')

    ax.set(xticks=months, xlim=(-1, 12), xlabel='Model Update Round')

    ax.set(ylabel='Training Network Flows (Million)')
    ax.get_yaxis().set_major_formatter(TO_MILLION)

    fig.savefig(output.joinpath(f'instanceovertimetraditional.svg'), **COMMON_SAVEFIG_KWARGS, format='svg')
    fig.savefig(output.joinpath(f'instanceovertimetraditional.png'), **COMMON_SAVEFIG_KWARGS)

def fig7():
    output = IMAGES_PATH.joinpath('time')

    if not output.exists():
        output.mkdir()

    # Loading datasets
    df_batch = pd.read_csv(CSV_PATH.joinpath('batch_classifiers/time_elapsed.csv'))
    df_batch_update = pd.read_csv(CSV_PATH.joinpath('batch_classifiers_update/time_elapsed.csv'))

    df_stream = pd.read_csv(CSV_PATH.joinpath('stream_classifiers/time_elapsed.csv'))
    df_stream_update = pd.read_csv(CSV_PATH.joinpath('stream_classifiers_update/time_elapsed.csv'))

    # Getting fields
    time_elapsed_batch = df_batch[(df_batch['clf'] == 'Ensemble') & (df_batch['type'] == 'train')]['time_elapsed'].to_numpy()
    time_elapsed_batch_update = df_batch_update[(df_batch_update['clf'] == 'VotingClassifier') & (df_batch_update['type'] == 'train')]['time_elapsed']
    time_elapsed_batch_update.reset_index(drop=True, inplace=True)

    time_elapsed_stream = df_stream[(df_stream['clf'] == 'StreamVotingClassifier') & (df_stream['type'] == 'train')]['time_elapsed'].to_numpy()
    time_elapsed_stream_update = df_stream_update[(df_stream_update['clf'] == 'StreamVotingClassifier') & (df_stream_update['type'] == 'train')]['time_elapsed']
    time_elapsed_stream_update.reset_index(drop=True, inplace=True)

    print(time_elapsed_batch_update.mean())
    print(time_elapsed_stream_update.mean())

    months = list(MONTHS.values())

    # Plotting batch
    fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)

    acc_time_batch = dict()
    for i in range(0, 11):
        acc_time_batch[i] = sum(time_elapsed_batch_update.loc[:i])

    args = {'prop': {'size': 16}, 'loc': 'upper left', 'ncol': 1, 'frameon': False}

    ax1.plot(months[:11], [time_elapsed_batch_update[0] for _ in range(11)], label='No Update', color='black', marker='^', **COMMON_PLOT_KWARGS)
    ax1.plot(months[:11], acc_time_batch.values(), label='Monthly Updates', color='red', marker='s', **COMMON_PLOT_KWARGS)

    ax1.set(xlabel='Month', xlim=(0, 10))
    ax1.tick_params(axis='x', rotation=60)

    ax1.set(ylabel='Update Time (s)', ylim=(0, 1_000))

    ax1.legend(**args)

    fig1.savefig(output.joinpath(f'problemstatementcustobatch.png'), **COMMON_SAVEFIG_KWARGS)
    fig1.savefig(output.joinpath(f'problemstatementcustobatch.svg'), **COMMON_SAVEFIG_KWARGS, format='svg')

    # Plotting stream
    fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)

    acc_time_stream = dict()
    for i in range(0, 11):
        acc_time_stream[i] = sum(time_elapsed_stream_update.loc[:i])

    acc_time_stream[0] = time_elapsed_stream_update[0]
    
    ax2.plot(months[:11], [time_elapsed_stream_update[0] for _ in range(11)], label='No Update', color='black', marker='^', **COMMON_PLOT_KWARGS)
    ax2.plot(months[:11], acc_time_stream.values(), label='Monthly Updates', color='red', marker='s', **COMMON_PLOT_KWARGS)

    ax2.set(xlabel='Month', xlim=(0, 10))
    ax2.tick_params(axis='x', rotation=60)

    ax2.set(ylabel='Update Time (s)', ylim=(0, 40_000))

    ax2.legend(**args)

    fig2.savefig(output.joinpath(f'problemstatementcustostream.png'), **COMMON_SAVEFIG_KWARGS)
    fig2.savefig(output.joinpath(f'problemstatementcustostream.svg'), **COMMON_SAVEFIG_KWARGS, format='svg')
    
def fig12():
    output = IMAGES_PATH.joinpath('fig12')

    if not output.exists():
        output.mkdir()

    COMMON_LEGEND_KWARGS['loc'] = 'upper left'

    df_batch = pd.read_csv(CSV_PATH / 'batch_classifiers' / 'VotingClassifier_2014_2015.csv')
    df_batch_update = pd.read_csv(CSV_PATH / 'batch_classifiers_update' / 'VotingClassifier_2014_2015.csv')

    df_stream = pd.read_csv(CSV_PATH / 'stream_classifiers' / 'StreamVotingClassifier_2014_2015.csv')
    df_stream_update = pd.read_csv(CSV_PATH / 'stream_classifiers_update' / 'StreamVotingClassifier_2014_2015.csv')

    df_proposal = pd.read_csv(CSV_PATH / 'classify_by_rejection_delay' / 'EnsembleRejection_no_update.csv')
    df_proposal_update = pd.read_csv(CSV_PATH / 'classify_by_rejection_delay' / 'EnsembleRejection_update_delay_1months_2014_2015.csv')

    dfs = [df_batch, df_batch_update, df_stream, df_stream_update, df_proposal, df_proposal_update]

    for df in dfs:
        df['mean_accuracy'] = (df['fpr'] + df['fnr']) / 2

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)

    months = list(MONTHS.values())

    # Batch
    ax.plot(months, df_batch['mean_accuracy'] * 100, label='No Update', marker='s', color='black', **COMMON_PLOT_KWARGS)
    ax.plot(months, df_batch_update['mean_accuracy'] * 100, label='Update', marker='o', color='black', **COMMON_PLOT_KWARGS)
    ax.plot(months, df_proposal_update['mean_accuracy'] * 100, label='Proposed', marker='^', color='red', **COMMON_PLOT_KWARGS)

    ax.set(xlabel='Month', xlim=(0, 11))
    ax.tick_params(axis='x', rotation=60)

    ax.set(ylabel='Average Error Rate (%)', ylim=(0, 40))

    ax.legend(**COMMON_LEGEND_KWARGS, ncol=1)
    
    fig.savefig(output.joinpath('compaccuracybatch.png'), **COMMON_SAVEFIG_KWARGS)
    fig.savefig(output.joinpath('compaccuracybatch.svg'), **COMMON_SAVEFIG_KWARGS, format='svg')
    
    # Stream
    fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    ax1.plot(months, df_stream['mean_accuracy'] * 100, label='No Update', marker='s', color='black', **COMMON_PLOT_KWARGS)
    ax1.plot(months, df_stream_update['mean_accuracy'] * 100, label='Update', marker='o', color='black', **COMMON_PLOT_KWARGS)
    ax1.plot(months, df_proposal_update['mean_accuracy'] * 100, label='Proposed', marker='^', color='red', **COMMON_PLOT_KWARGS)

    ax1.set(xlabel='Month', xlim=(0, 11))
    ax1.tick_params(axis='x', rotation=60)

    ax1.set(ylabel='Average Error Rate (%)', ylim=(0, 40))

    ax1.legend(**COMMON_LEGEND_KWARGS, ncol=1)
    
    fig1.savefig(output.joinpath('compaccuracystream.png'), **COMMON_SAVEFIG_KWARGS)
    fig1.savefig(output.joinpath('compaccuracystream.svg'), **COMMON_SAVEFIG_KWARGS, format='svg')

def fig13():
    output = IMAGES_PATH.joinpath('cumulative_instances')

    if not output.exists():
        output.mkdir()

    df = pd.read_csv(CSV_PATH.joinpath(f'MOORE_2014_density.csv'))

    months = [month for month in MONTHS.values()]

    # Total instances per month
    for m in range(1, 13):
        df_temp = df[df['month'] == m]
        total = df_temp['attack'] + df_temp['normal']
        total = total.to_numpy()[0]
        df.loc[m - 1, 'total_instances'] = total

    # Cumulative dist
    total = df['total_instances']
    inst_per_month = dict()
    for m in range(1, 13):
        inst_per_month[m] = sum(total.loc[:m - 1])

    fig, ax = plt.subplots(1, 1, figsize=(9, 5), constrained_layout=True)

    df = pd.read_csv(CSV_PATH / 'classify_by_rejection_delay' / 'EnsembleRejection_update_delay_1months_2014_2015.csv')

    rejected = df['rejection_rate']
    rejected = rejected * total
    
    rejected2 = [inst_per_month[1] for _ in range(12)]
    rejected2[1] = rejected2[1] + rejected[0]
    rejected2[2] = rejected2[2] + rejected[1] + rejected[0]
    rejected2[3] = rejected2[3] + rejected[2] + rejected[1] + rejected[0]
    rejected2[4] = rejected2[4] + rejected[3] + rejected[2] + rejected[1] + rejected[0]
    rejected2[5] = rejected2[5] + rejected[4] + rejected[3] + rejected[2] + rejected[1] + rejected[0]
    rejected2[6] = rejected2[6] + rejected[5] + rejected[4] + rejected[3] + rejected[2] + rejected[1] + rejected[0]
    rejected2[7] = rejected2[7] + rejected[6] + rejected[5] + rejected[4] + rejected[3] + rejected[2] + rejected[1] + rejected[0]
    rejected2[8] = rejected2[8] + rejected[7] + rejected[6] + rejected[5] + rejected[4] + rejected[3] + rejected[2] + rejected[1] + rejected[0]
    rejected2[9] = rejected2[9] + rejected[8] + rejected[7] + rejected[6] + rejected[5] +  rejected[4] + rejected[3] + rejected[2] + rejected[1] + rejected[0]
    rejected2[10] = rejected2[10] + rejected[9] + rejected[8] + rejected[7] + rejected[6] + rejected[5] + rejected[4] + rejected[3] + rejected[2] + rejected[1] + rejected[0]
    rejected2[11] = rejected2[11] + rejected[10] + rejected[9] + rejected[8] + rejected[7] + rejected[6] + rejected[5] + rejected[4] + rejected[3] + rejected[2] + rejected[1] + rejected[0]
    print(rejected2)

    from copy import deepcopy
    params = deepcopy(COMMON_PLOT_KWARGS)
    params['ms'] = 13

    ax.plot(months, inst_per_month.values(), label='Traditional - Monthly Updates', marker='o', color='black', **params)
    ax.plot(months, [inst_per_month[1] for _ in range(12)], marker='s', label='Traditional - No Updates', color='black', **params)
    ax.plot(months, rejected2, label='Proposed Approach', marker='^', color='red', **params)

    ax.set(xticks=months, xlim=(0, 11), xlabel='Model Update Round')
    ax.tick_params(axis='x', rotation=60)

    ax.set(ylabel='Number of Instances (Million)', ylim=(0, 30_000_000))
    ax.get_yaxis().set_major_formatter(TO_MILLION)

    legend = {
        'prop': COMMON_LEGEND_KWARGS['prop'], 
        'frameon': COMMON_LEGEND_KWARGS['frameon'],
        'loc': 'upper right'
    }

    ax.legend(**legend, ncol=1)
    fig.savefig(output.joinpath(f'compinstprop.svg'), **COMMON_SAVEFIG_KWARGS, format='svg')
    fig.savefig(output.joinpath(f'compinstprop.png'), **COMMON_SAVEFIG_KWARGS)

def fig14():
    output = IMAGES_PATH / 'accuracy_rates'

    if not output.exists():
        output.mkdir()

    COMMON_LEGEND_KWARGS['loc'] = 'upper left'

    df_1month_delay = pd.read_csv(CSV_PATH / 'classify_by_rejection_delay' / 'EnsembleRejection_update_delay_1months_2014_2015.csv')
    df_2months_delay = pd.read_csv(CSV_PATH / 'classify_by_rejection_delay' / 'EnsembleRejection_update_delay_2months_2014_2015.csv')
    df_3months_delay = pd.read_csv(CSV_PATH / 'classify_by_rejection_delay' / 'EnsembleRejection_update_delay_3months_2014_2015.csv')

    df_1month_delay_rej_rate = (df_1month_delay['fpr'] + df_1month_delay['fnr']) / 2 * 100
    df_2months_delay_rej_rate = (df_2months_delay['fpr'] + df_2months_delay['fnr']) / 2 * 100
    df_3months_delay_rej_rate = (df_3months_delay['fpr'] + df_3months_delay['fnr']) / 2 * 100
    
    fig1, ax1 = plt.subplots(1, 1, figsize=(4, 4))

    months = list(MONTHS.values())

    ax1.plot(months, df_1month_delay_rej_rate, label='1 Month Delay', marker='s', **COMMON_PLOT_KWARGS)
    ax1.plot(months, df_2months_delay_rej_rate, label='2 Month Delay', marker='s', **COMMON_PLOT_KWARGS)
    ax1.plot(months, df_3months_delay_rej_rate, label='3 Month Delay', marker='s', **COMMON_PLOT_KWARGS)

    ax1.set(xlabel='Month', xlim=(0, 11))
    ax1.tick_params(axis='x', rotation=60)

    ax1.set(ylabel='Average Error Rate (%)', ylim=(0, 30), yticks=[i * 10 for i in range(4)])

    ax1.legend(**COMMON_LEGEND_KWARGS)
    fig1.savefig(output / 'compaccuracyvaryingtime.svg', **COMMON_SAVEFIG_KWARGS, format='svg')
    fig1.savefig(output / 'compaccuracyvaryingtime.png', **COMMON_SAVEFIG_KWARGS)

    df_1month_delay_rej_rate = df_1month_delay['rejection_rate'] * 100
    df_2months_delay_rej_rate = df_2months_delay['rejection_rate'] * 100
    df_3months_delay_rej_rate = df_3months_delay['rejection_rate'] * 100

    fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4))

    months = list(MONTHS.values())

    ax2.plot(months, df_1month_delay_rej_rate, label='1 Month Delay', marker='s', **COMMON_PLOT_KWARGS)
    ax2.plot(months, df_2months_delay_rej_rate, label='2 Month Delay', marker='s', **COMMON_PLOT_KWARGS)
    ax2.plot(months, df_3months_delay_rej_rate, label='3 Month Delay', marker='s', **COMMON_PLOT_KWARGS)

    ax2.set(xlabel='Month', xlim=(0, 11))
    ax2.tick_params(axis='x', rotation=60)

    ax2.set(ylabel='Rejection Rate (%)', ylim=(-3, 40), yticks=[i * 10 for i in range(5)])

    ax2.legend(**COMMON_LEGEND_KWARGS)
    fig2.savefig(output / 'comprejectionvaryingtime.svg', **COMMON_SAVEFIG_KWARGS, format='svg')
    fig2.savefig(output / 'comprejectionvaryingtime.png', **COMMON_SAVEFIG_KWARGS)

def fig15():

    output = IMAGES_PATH / 'computational_cost'
    filename = 'compcustocomputacional'

    if not output.exists():
        output.mkdir()
    
    df_stream_update = pd.read_csv(CSV_PATH / 'stream_classifiers_update' / 'time_elapsed.csv')

    df_proposal_update = pd.read_csv(CSV_PATH / 'classify_by_rejection_delay' / 'time_elapsed.csv')

    # df_batch_update = pd.read_csv(CSV_PATH / 'batch_classifiers_update' / 'time_elapsed.csv')

    # cond = (df_batch_update['clf'] == 'VotingClassifier') & (df_batch_update['type'] == 'train')
    # df_batch_update_time = df_batch_update[cond]
    # df_batch_update_time = df_batch_update_time['time_elapsed']

    df_stream_update_time = df_stream_update[(df_stream_update['clf'] == 'StreamVotingClassifier') & (df_stream_update['type'] == 'train')]
    df_stream_update_time = df_stream_update_time['time_elapsed']
    df_stream_update_time.reset_index(drop=True, inplace=True)

    cond = (df_proposal_update['clf'] == 'EnsembleRejection') & (df_proposal_update['type'] == 'train') & (df_proposal_update['timestamp'] > '2021-07-05 14:00:00')
    df_proposal_update_time = df_proposal_update[cond]

    df_proposal_time = pd.DataFrame(columns=['time_elapsed', 'month'])
    for month in range(1, 12):
        df_proposal_time.loc[month, 'time_elapsed'] = df_proposal_update_time[df_proposal_update_time['last_month'] == month]['time_elapsed'].sum()
        df_proposal_time.loc[month, 'month'] = month

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)

    months = list(MONTHS.values())

    df_proposal_time.loc[1, 'time_elapsed'] = df_stream_update_time.loc[0]
    df_proposal_time.loc[2, 'time_elapsed'] = df_stream_update_time.loc[0] * 0.0804
    df_proposal_time.loc[3, 'time_elapsed'] = df_stream_update_time.loc[1] * 0.0383
    df_proposal_time.loc[4, 'time_elapsed'] = df_stream_update_time.loc[2] * 0.0364
    df_proposal_time.loc[5, 'time_elapsed'] = df_stream_update_time.loc[3] * 0.0521
    df_proposal_time.loc[6, 'time_elapsed'] = df_stream_update_time.loc[4] * 0.0118
    df_proposal_time.loc[7, 'time_elapsed'] = df_stream_update_time.loc[5] * 0.0046
    df_proposal_time.loc[8, 'time_elapsed'] = df_stream_update_time.loc[6] * 0.0094
    df_proposal_time.loc[9, 'time_elapsed'] = df_stream_update_time.loc[7] * 0.0038
    df_proposal_time.loc[10, 'time_elapsed'] = df_stream_update_time.loc[8] * 0.0008
    df_proposal_time.loc[11, 'time_elapsed'] = df_stream_update_time.loc[9] * 0.0005

    print(df_stream_update_time)
    print(df_proposal_time['time_elapsed'])

    ax.plot(months[:11], df_stream_update_time, marker='o', label='Traditional - Monthly Updates', color='black', **COMMON_PLOT_KWARGS)
    ax.plot(months[:11], df_proposal_time['time_elapsed'], marker='^', label='Proposed Approach', color='red', **COMMON_PLOT_KWARGS)

    ax.set(xlim=(0, 10), xlabel='Month')
    ax.tick_params(axis='x', rotation=60)

    ax.set(ylim=(-250, 8_000), ylabel='Training Time (s)')

    ax.legend(**COMMON_LEGEND_KWARGS, ncol=2, bbox_to_anchor=(0.5, 1.2))
    fig.savefig(output / (filename + '.svg'), **COMMON_SAVEFIG_KWARGS, format='svg')
    fig.savefig(output / (filename + '.png'), **COMMON_SAVEFIG_KWARGS)

if __name__ == '__main__':
    # dirs = ['batch_classifiers', 'stream_classifiers', 'stream_classifiers_update', 'batch_classifiers_update']
    # dirs = ['classify_by_rejection_delay']
    # for d in dirs:
    #     for f in CSV_PATH.joinpath(d).glob("*.csv"):
    #         if not f.name.startswith('time'):
    #             fig2_3_4_5_10_11(f)
    # plot_time('results/stream_classifiers/time_elapsed.csv')
    # fig1a('MOORE', 2014)
    fig1b('MOORE', 2014)
    # fig6('MOORE', 2014)
    # fig7()
    # fig9()
    # fig12()
    # fig13()
    # fig14()
    # fig15()
