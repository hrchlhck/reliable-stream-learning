from pathlib import Path

__all__ = ['BASES', 'DATASETS', 'MONTHS']

BASES = ["NIGEL", "MOORE", "VIEGAS", "ORUNADA"]
DATASETS_PATH = Path("./outDayDataset/")
DATASETS = {year: {base : DATASETS_PATH.joinpath(str(year)).joinpath(base) for base in BASES} 
            for year in range(2010, 2017)}
MONTHS_NAME = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS = [(str(i)) if i >= 10 else f"0{i}" for i in range(1, 13)]
MONTHS = list(zip(MONTHS, MONTHS_NAME))
