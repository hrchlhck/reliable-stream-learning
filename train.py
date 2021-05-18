from constants import MONTHS

from utils import (
    get_files, get_X_y, 
    clf_predict, show_results
)

from log import get_logger

from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import OzaBaggingClassifier
from skmultiflow.lazy import KNNClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Process
from threading import Thread, Lock  
from pathlib import Path
from csv import DictWriter

LOGGER = get_logger(Path(__file__).name)

get_cls_name = lambda obj: obj.__class__.__name__

def _test_model(clf: object, _from: int, _to: int, base: str, filename: Path, mutex: Lock):
    # Test January of 'test_year' with all subsequent month by 'test_year'
    for test_year in range(_from, _to):
        LOGGER.info(f"Started testing {get_cls_name(clf)} from {test_year} to {_to - 1}")

        for m, mn in MONTHS:
            LOGGER.info(f"Started testing {get_cls_name(clf)} at {mn}/{test_year}")
            
            # Get files for 'test_year/BASE/m'
            test_files = get_files(test_year, base, m)
            
            # Test and save results
            result_p_month = clf_predict(clf, test_files['files'], LOGGER)
            
            # Parsing results
            clf_metrics = show_results(result_p_month)
            clf_metrics['month'] = mn
            clf_metrics['test_year'] = test_year

            # Check if file exists and change mode
            mode = 'w'
            if filename.exists():
                mode = 'a' 

            mutex.acquire()
            # Save clf_metrics into a CSV
            with open(filename, mode) as fd:
                header = clf_metrics.keys()
                
                writer = DictWriter(fd, fieldnames=header)

                if mode == 'w':
                    writer.writeheader()

                writer.writerow(clf_metrics)
                LOGGER.debug(clf_metrics)
            mutex.release()

def test(_from: int, _to: int, base: str, classifier: object):
    filename = Path(f"results/{get_cls_name(classifier)}_{_from}_{_to - 1}_{base}.csv")

    if not filename.parent.exists():
        filename.parent.mkdir()

    LOGGER.info(f"Metrics will be saved at results/{filename.name}")
    
    # Get files for '_from/BASE/01'
    train_files = get_files(_from, base, '01')
    
    # Get first day of first month for partial fit
    X, y = get_X_y(train_files['files'][0])

    # Partial fit with first instance within a list
    if not isinstance(classifier, RandomForestClassifier):
        classifier = classifier.partial_fit([X[0]], [y[0]], classes=[0, 1])
    
    # Train the classifier with the remaining files
    LOGGER.info(f"Started training {get_cls_name(classifier)} for year {_from}")
    LOGGER.debug(f"Classifier parameters: {classifier.get_params()}")
    for f in train_files['files'][1:]:
        X, y = get_X_y(f)
        LOGGER.debug(f"Training {get_cls_name(classifier)} with file {f}")
        if not isinstance(classifier, RandomForestClassifier):
            for i in range(len(X)):
                classifier = classifier.partial_fit([X[i]], [y[i]])
        else:
            classifier = classifier.fit(X, y)
        LOGGER.debug(f"Finished training {get_cls_name(classifier)} with file {f}")
    LOGGER.info(f"Done training {get_cls_name(classifier)} for year {_from}")

    for train_year in range(_from, _to):
        mutex = Lock()
        Thread(target=_test_model, args=(classifier, train_year, _to, base, filename, mutex)).start()
        

if __name__ == '__main__':
    BASE = 'VIEGAS'
    ESTIMATOR = KNNClassifier()
    FROM = 2010
    TO = 2012
    CLASSIFIERS = [
        OzaBaggingClassifier(random_state=42),
        AdaptiveRandomForestClassifier(),
        RandomForestClassifier(n_estimators=100, max_depth=4, criterion='entropy'),
        HoeffdingTreeClassifier()
    ]

    processes = [Process(target=test, args=(FROM, TO, BASE, c)) for c in CLASSIFIERS] 

    try:    
        for p in processes:
            p.start()
            LOGGER.info(f"Started process {p}")
    except KeyboardInterrupt:
        for p in processes:
            p.kill()