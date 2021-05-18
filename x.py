from constants import MONTHS
from csv import DictReader
from matplotlib import pyplot as plt
from pathlib import Path
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python x.py CSV FILE")
        exit(1)
    
    filename = Path(sys.argv[1])
    output_dir = Path('images/')

    if not output_dir.exists():
        output_dir.mkdir()

    with open(filename, moreaderde='r') as fd:
        reader = DictReader(fd)
        reader = list(reader)

        fpr = []
        fnr = []
        months = list(map(lambda x: x[1], MONTHS[1:]))

        for i in range(len(MONTHS[1:])):
            month = MONTHS[i+1][1]
            fpr.append(reader[i]['fpr'])
            fnr.append(reader[i]['fnr'])

    fpr = list(map(lambda x: float(x), fpr))
    fnr = list(map(lambda x: float(x), fnr))

    year_range = filename.name.split('_')[1:3]
    
    plt.plot(months, fpr, marker='.', label='FPR', color='red')
    plt.plot(months, fnr, marker='*', label='FNR', color='black')
    plt.grid(True)
    plt.legend()
    plt.title(f'FN Rate vs FP Rate over time for interval {year_range[0]} - {year_range[1]}')
    plt.savefig(f"images/{filename.name[:-4]}.png", dpi=210)
    plt.show()
