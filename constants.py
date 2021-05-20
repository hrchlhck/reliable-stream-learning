from pathlib import Path

__all__ = ['VIEWS', 'DATASETS', 'MONTHS', 'CSV_PATH', 'IMAGES_PATH', 'LOGS_PATH']

VIEWS = ["NIGEL", "MOORE", "VIEGAS", "ORUNADA"]
DATASETS_PATH = Path("./outDayDataset/")
DATASETS = {year: {view : DATASETS_PATH.joinpath(str(year)).joinpath(view) for view in VIEWS} 
            for year in range(2010, 2017)}
MONTHS_NAME = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS = [(str(i)) if i >= 10 else f"0{i}" for i in range(1, 13)]
MONTHS = list(zip(MONTHS, MONTHS_NAME))
CSV_PATH = Path('results/')
IMAGES_PATH = Path('images/')
LOGS_PATH = Path('logs/')
MODELS_PATH = Path('models/')