from ensemble_rejection import EnsembleRejection
from x import plot_pareto, plot_results
from constants import MONTHS, CSV_PATH, MODELS_PATH

from utils import (
    compute_pareto, ensemble_predict, get_files, get_X_y, 
    clf_predict, get_operation_point, 
    print_progress, save_csv, save_model, 
    show_results, timer, 
    load_model, ensemble,
    get_metrics, get_cls_name,
)

from log import get_logger

from skmultiflow.meta import LeveragingBaggingClassifier
from skmultiflow.meta import OzaBaggingClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.data import DataStream
from sklearn.ensemble import RandomForestClassifier
from threading import Thread
from pathlib import Path
from csv import DictWriter
from random import Random
from datetime import datetime
import pandas as pd
DATE = datetime.now().strftime('%d_%m_%Y-%H-%M')
LOGGER = get_logger(f'train_moore_2014_{DATE}')

# For r.choice
r = Random(1)

normalize = lambda x, max, min: (x - min) / (max - min) 

def _test_ensemble(classifiers: list, _from: int, _to: int, view: str, filename: Path, files: list):
    for test_year in range(_from, _to):
        LOGGER.info(f"Started testing ensemble of classifiers {classifiers} from {test_year} to {_to - 1}")

        for m, mn in MONTHS:
            LOGGER.info(f"Started testing ensemble at {mn}/{test_year}")
            
            # Get files by month
            test_files = files[m]

            # Predicting
            result_p_month = ensemble(classifiers, test_files, LOGGER)

            # Parsing metrics
            clf_metrics = show_results(result_p_month)
            clf_metrics['month'] = mn
            clf_metrics['test_year'] = test_year

            save_csv(filename, clf_metrics, logger=LOGGER)

@timer(logger=LOGGER)
def _test_model(clf: object, _from: int, _to: int, base: str, filename: Path, files: list):
    # Test January of 'test_year' with all subsequent month by 'test_year'
    clf_name = get_cls_name(clf)

    if isinstance(filename, str):
        filename = Path(filename)

    for test_year in range(_from, _to):
        LOGGER.info(f"Started testing {clf_name} from {test_year} to {_to - 1}")

        for m, mn in MONTHS:
            LOGGER.info(f"Started testing {clf_name} at {mn}/{test_year}")
            
            # Get files for 'test_year/BASE/m'
            # test_files = get_files(test_year, base, m)
            test_files = files[m]
            
            # Test and save results
            # result_p_month = clf_predict(clf, test_files['files'], LOGGER)
            result_p_month = clf_predict(clf, test_files, LOGGER)
            
            # Parsing results
            clf_metrics = show_results(result_p_month)
            clf_metrics['month'] = mn
            clf_metrics['test_year'] = test_year

            save_csv(filename, clf_metrics, logger=LOGGER)

@timer(logger=LOGGER)
def test(_from: int, _to: int, base: str, classifier: object, files: list):
    clf_name = get_cls_name(classifier)

    output_name = f"{clf_name}_{_from}_{_to - 1}_{base}_{DATE}.csv"
    file_path = CSV_PATH.joinpath(output_name)

    if not CSV_PATH.exists():
        CSV_PATH.mkdir()

    LOGGER.info(f"Metrics will be saved at results/{file_path.name}")
    
    # Get files for '_from/BASE/01'
    train_files = get_files(_from, base, '01')
    days = len(train_files['files'])

    # Get first day of first month for partial fit
    X, y = get_X_y(train_files['files'][0])

    # Partial fit with first instance within a list
    if not isinstance(classifier, RandomForestClassifier):
        classifier = classifier.partial_fit([X[0]], [y[0]], classes=[0, 1])
    
    # Train the classifier with the remaining files
    LOGGER.info(f"Started training {clf_name} for year {_from}")
    LOGGER.debug(f"Classifier parameters: {classifier}")

    for f in train_files['files'][1: int(days / 2)]:
        X, y = get_X_y(f)
        LOGGER.debug(f"Training {clf_name} with file {f}")
        
        if isinstance(classifier, RandomForestClassifier):
            classifier = classifier.fit(X, y)
        else:
            ds = DataStream(X, y=y, n_targets=2)
            max_samples = len(X)
            samples = 0
            while ds.has_more_samples() and samples < max_samples:
                classifier = classifier.partial_fit(X, y)
                X, y = ds.next_sample()
                samples += 1
        LOGGER.debug(f"Finished training {clf_name} with file {f}")
    LOGGER.info(f"Done training {clf_name} for year {_from}")
    
    # Saving the classifier
    save_model(classifier, clf_name)

    LOGGER.info(f"Saved {clf_name} at {MODELS_PATH.name}/{clf_name}.model")

    # _test_model(classifier, _from, _to, base, file_path, files)

    # # Generate a plot with FPR and FNR results
    # LOGGER.info(f'Generating plot for {output_name}')
    # plot_results(file_path)


def main_classifiers():
    if not MODELS_PATH.exists():
        MODELS_PATH.mkdir()

    VIEWS = ['MOORE']
    ESTIMATOR = HoeffdingTreeClassifier
    N_DAYS = 7
    N_ESTIMATORS = 3
    FROM = 2014
    TO = 2015
    files = {month: [r.choice(get_files(FROM, VIEWS[0], month)['files']) for _ in range(N_DAYS)] for month, _ in MONTHS}
    CLASSIFIERS = [load_model(clf.name) for clf in MODELS_PATH.glob("*.model")]

    threads = list()

    for view in VIEWS:
        for clf in CLASSIFIERS:
            args = (clf, FROM, TO, view, f"{get_cls_name(clf)}_{FROM}_{TO}_{DATE}.csv", files)
            thread = Thread(target=_test_model, args=args)
            LOGGER.debug(f"Created thread {thread} for view {view} with classifier {clf}")
            threads.append(thread)

    try:
        for t in threads:
            t.start()
            LOGGER.debug(f"Started thread {t}")
    except KeyboardInterrupt:
        pass

def main_ensemble():
    VIEWS = ['MOORE']
    CLASSIFIERS = [(clf.name.split('.')[0], load_model(str(clf.name))) for clf in Path('models/').glob('*.model')]
    FROM = 2014
    TO = 2015
    N_DAYS = 7
    files = {month: [r.choice(get_files(FROM, VIEWS[0], month)['files'], ) for _ in range(N_DAYS)] for month, _ in MONTHS}
    OUTPUT = Path(f'results/ensemble_{FROM}_{TO}_{VIEWS[0]}.csv')
    
    _test_ensemble(CLASSIFIERS, FROM, TO, VIEWS[0], OUTPUT, files)

@timer(logger=LOGGER)
def main_rejection(clf):
    VIEWS = ['MOORE']
    FROM = 2014
    COLUMNS = ['true_class', 'predicted_class', 'normal_confidence', 'attack_confidence']
    CLF = load_model(clf)
    OUTPUT = CSV_PATH.joinpath('pareto', get_cls_name(CLF) + ".csv")

    if not OUTPUT.exists():
        files = get_files(FROM, VIEWS[0], '02')['files']

        df = pd.DataFrame(columns=COLUMNS)

        for count, file in enumerate(files):
            LOGGER.info(f"Testing file {file.name}")
            LOGGER.info(f"{count} out of {len(files)} files done")

            X, y = get_X_y(file)

            pred = CLF.predict(X)
            pred_proba = CLF.predict_proba(X)

            for i in range(len(X)):
                true = y[i]
                predicted = pred[i]
                normal = pred_proba[i, 0]
                attack = pred_proba[i, 1]
                df.loc[len(df)] = [true, predicted, normal, attack]
                print_progress(i, len(X), label=get_cls_name(CLF))
            df.to_csv(OUTPUT, index=False)

def _test_rejection(classifiers: object, rejection_table: dict, files: list, _from: int, _to: int, output_path: Path, logger):
    clf_name = "EnsembleRejection2_28_05__01"
    filename = output_path.joinpath(f"{clf_name}_{_from}_{_to}.csv")
    
    last_month = '01'
    actual_month = '01'
    er = EnsembleRejection(classifiers, rejection_table, [0, 1], logger=logger)
    for test_year in range(_from, _to):
        for m, mn in MONTHS:
            results = dict()
            actual_month = m
            for fi in range(len(files[m])):
                file = files[m][fi]
                logger.info(f"Testing file {file.name} for ensemble of {', '.join(er.names)}. Tested {fi} out of {len(files[m])} files")
                X, y = get_X_y(file)
                results[file.name] = er.predict(X, y, last_month=last_month, actual_month=actual_month, file_num=fi)

            # Parsing results
            ensemble_metrics = show_results(results)
            ensemble_metrics['month'] = mn
            ensemble_metrics['test_year'] = test_year

            save_csv(filename, ensemble_metrics, logger=LOGGER)
            last_month = actual_month

@timer(logger=LOGGER)
def main_classify_rejection():
    VIEWS = ['MOORE']
    FROM = 2014
    TO = 2015
    CLASSIFIERS = [load_model(clf.name) for clf in MODELS_PATH.glob('*.model')]
    OUTPUT = CSV_PATH.joinpath('classify_by_rejection')
    CLASSIFIER_THRESHOLDS_FILES = list(CSV_PATH.joinpath("pareto_computed").glob("*.csv"))
    CLASSIFIER_THRESHOLDS = get_operation_point(CLASSIFIER_THRESHOLDS_FILES, 'error_rate', 0.05)
    N_DAYS = 7

    if not OUTPUT.exists():
        OUTPUT.mkdir()

    files = {month: [r.choice(get_files(FROM, VIEWS[0], month)['files']) for _ in range(N_DAYS)] for month, _ in MONTHS}

    _test_rejection(CLASSIFIERS, CLASSIFIER_THRESHOLDS, files, FROM, TO, OUTPUT, LOGGER)

if __name__ == '__main__':
    # main_classifiers()
    # main_ensemble()
    # for clf in MODELS_PATH.glob('*.model'):
    #     main_rejection(clf.name)
    main_classify_rejection()

    # for file in CSV_PATH.joinpath('pareto').glob('*.csv'):
    #     if 'Classifier' in file.name:
    #         get_rejection_metrics(file)
    
    # for file in CSV_PATH.joinpath('rejection_metrics').glob('*.csv'):
    #     if 'Classifier' in file.name:
    #         compute_pareto(file)

    # plot_results()