from sys import argv
from constants import MONTHS

from utils import (
    get_files, get_X_y, 
    clf_predict, show_results
)

from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import OzaBaggingClassifier
from skmultiflow.lazy import KNNClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
from multiprocessing import Process
from threading import Thread
from pathlib import Path
from addict import Dict
from csv import DictWriter


def _test_model(clf: object, _from: int, _to: int, base: str, filename: Path):
    # Test January of 'test_year' with all subsequent month by 'test_year'
    for test_year in range(_from, _to):
        print(f"Started testing from {test_year} to {_to}")
        
        for m, mn in MONTHS[1:]:
            print(f"Started testing month {mn} from {test_year}")
            
            # Get files for 'test_year/BASE/m'
            test_files = get_files(test_year, base, m)
            
            # Test and save results
            result_p_month = clf_predict(clf, test_files['files'])
            
            # Parsing results
            clf_metrics = show_results(result_p_month)
            clf_metrics['month'] = mn
            clf_metrics['test_year'] = test_year

            # # Storing the results
            # results[_from][test_year][m] = clf_metrics

            # Check if file exists and change mode
            mode = 'w'
            if filename.exists():
                mode = 'a' 

            # Save clf_metrics into a CSV
            with open(filename, mode) as fd:
                header = clf_metrics.keys()
                
                writer = DictWriter(fd, fieldnames=header)

                if mode == 'w':
                    writer.writeheader()

                writer.writerow(clf_metrics)

def test(_from: int, _to: int, base: str):
    for train_year in range(_from, _to):
        filename = Path(f"results/{_from}_{_to}_{base}.csv")

        if not filename.parent.exists():
            filename.parent.mkdir()

        print(f"Metrics will be saved at results/{filename.name}")

        clf = HoeffdingTreeClassifier()
        
        # Get files for 'train_year/BASE/01'
        train_files = get_files(train_year, base, '01')
        
        # Get first day of first month for partial fit
        X, y = get_X_y(train_files['files'][0])

        # Partial fit with first instance within a list
        clf = clf.partial_fit([X[0]], [y[0]], classes=[0, 1])
        
        # Train the classifier with the remaining files
        print(f"Started training {clf.__class__.__name__} for year {train_year}")
        for f in train_files['files'][1:]:
            X, y = get_X_y(f)
            for i in range(len(X)):
                clf = clf.partial_fit([X[i]], [y[i]])
        print(f"Done training {clf.__class__.__name__} for year {train_year}")

        Thread(target=_test_model, args=(clf, train_year, _to, base, filename)).start()
        

if __name__ == '__main__':
    BASE = 'VIEGAS'
    processes = [
        Process(target=test, args=(2010, 2012, BASE)),
        Process(target=test, args=(2011, 2013, BASE)),
        Process(target=test, args=(2012, 2014, BASE)),
    ]
    
    for p in processes:
        p.start()
        print(f"Started process {p}")