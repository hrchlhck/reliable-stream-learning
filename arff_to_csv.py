from utils import toCsv
from pathlib import Path

dirs = Path('outDayDataset/unbalanced/2014/MOORE')

for month_dir in dirs.glob('*'):
    print(month_dir.name)
    for file in month_dir.glob('*.arff'):
        with open(file, 'r') as _in:
            c = _in.readlines()
            data = toCsv(c)
            with open(file.parent / f'{file.name[:-5]}.csv', 'w') as out:
                out.writelines(data)