from ensemble_rejection import EnsembleRejection, StreamVotingClassifier
from x import plot_pareto, plot_results
from constants import MONTHS, CSV_PATH, MODELS_PATH

from utils import (
    compute_pareto, ensemble_predict, get_files, get_X_y, 
    clf_predict, get_operation_point, 
    print_progress, save_csv, save_model, 
    show_results, timer, 
    load_model, ensemble,
    get_metrics, get_cls_name
)

from log import get_logger

from skmultiflow.meta import LeveragingBaggingClassifier
from skmultiflow.meta import OzaBaggingClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.data import DataStream

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)

from threading import Thread, Lock
from pathlib import Path
from random import Random
from datetime import datetime
from time import perf_counter
from copy import deepcopy

import pandas as pd


DATE = datetime.now().strftime('%d_%m_%Y-%H-%M')
LOGGER = get_logger(f'train_moore_2014_{DATE}')

# For r.choice
r = Random(1)

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

    LOGGER.info(f"Metrics will be saved at resufileslts/{file_path.name}")
    
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


def main_classifiers(update=False):
    VIEW = 'MOORE'
    FROM = 2014
    TO = 2015
    N_DAYS = 7
    ESTIMATOR = HoeffdingTreeClassifier()
    params = {'base_estimator': ESTIMATOR, 'n_estimators': 3}
    CLASSIFIERS = [
        HoeffdingTreeClassifier(), 
        LeveragingBaggingClassifier(**params), 
        OzaBaggingClassifier(**params),
    ]
    CLASSIFIERS.append(StreamVotingClassifier(['lev', CLASSIFIERS[1]], ['oza', CLASSIFIERS[2]], ['hoeff', CLASSIFIERS[0]], logger=LOGGER))
    OUTPUT = CSV_PATH.joinpath('stream_classifiers_update')
    FILES = {month: [r.choice(get_files(FROM, VIEW, month)['files']) for _ in range(N_DAYS)] for month, _ in MONTHS}

    if not OUTPUT.exists():
        OUTPUT.mkdir()
        
    def test_classifier(classifier: list, files: dict, mutex: Lock, update=False):
        name = get_cls_name(classifier)
        
        # Training
        start = perf_counter()
        for i, file in enumerate(FILES['01']):
            LOGGER.info(f"Started training with day {i + 1} the classifier {name}")
            X, y = get_X_y(file)
            ds = DataStream(X, y=y, n_targets=2)
            max_samples = len(X)
            samples = 0
            classifier = classifier.partial_fit([X[0]], [y[0]], classes=[0, 1])
            while ds.has_more_samples() and samples < max_samples:
                X, y = ds.next_sample()
                classifier = classifier.partial_fit(X, y, classes=[0, 1])
                samples += 1
        end = perf_counter() - start
        times = {'clf': name, 'time_elapsed': end, 'type': 'train', 'month': '01'}
        save_csv(OUTPUT.joinpath('time_elapsed.csv'), times, logger=LOGGER)

        # Testing        
        last_month = '01'
        actual_month = '01'

        for month in files:
            actual_month = month
            _files = files[month]

            if update and actual_month != last_month and actual_month != '02':
                LOGGER.debug(f"actual_month: {actual_month}, last_month: {last_month}")
                LOGGER.info(f"Started updating classifier {name} with month {last_month}")

                last_month_files = files[last_month]

                start_train = perf_counter()
                for i, file in enumerate(last_month_files):
                    LOGGER.info(f"Started updating at day {i + 1} the classifier {name} with month {last_month}")
                    X, y = get_X_y(file)
                    ds = DataStream(X, y=y, n_targets=2)
                    max_samples = len(X)
                    samples = 0
                    classifier = classifier.partial_fit([X[0]], [y[0]], classes=[0, 1])
                    while ds.has_more_samples() and samples < max_samples:
                        X, y = ds.next_sample()
                        classifier = classifier.partial_fit(X, y, classes=[0, 1])
                        samples += 1
                end_train = perf_counter() - start_train
                times = {'clf': name, 'time_elapsed': end_train, 'type': 'train', 'month': last_month}
                mutex.acquire()
                save_csv(OUTPUT.joinpath('time_elapsed.csv'), times, logger=LOGGER)
                mutex.release()

            last_month = month
            LOGGER.info(f"Started testing {name} at month {month}")
            start = perf_counter()
            _test_batch(classifier, _files, OUTPUT, month, FROM, TO)
            end = perf_counter() - start
            times = {'clf': name, 'time_elapsed': end, 'type': 'test', 'month': month}
            mutex.acquire()
            save_csv(OUTPUT.joinpath('time_elapsed.csv'), times, logger=LOGGER)
            mutex.release()
    
    mutex = Lock()
    for clf in CLASSIFIERS:
        Thread(target=test_classifier, args=(clf, FILES, mutex, update)).start()

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

def _test_rejection(classifiers: object, rejection_table: dict, files: list, _from: int, _to: int, output_path: Path, logger, update: bool, mutex: Lock):
    clf_name = "EnsembleRejection2_31_05__1_single_update"
    filename = output_path.joinpath(f"{clf_name}_{_from}_{_to}.csv")
    output_day = output_path.joinpath(f'per_day_{clf_name}')

    if not output_day.exists():
        output_day.mkdir()

    last_month = '01'
    actual_month = '01'
    er = EnsembleRejection(classifiers, rejection_table, [0, 1], logger=logger)
    for test_year in range(_from, _to):
        for m, mn in MONTHS:
            results = dict()
            actual_month = m

            start = perf_counter()
            for fi in range(len(files[m])):
                file = files[m][fi]
                logger.info(f"Testing file {file.name} for ensemble of {', '.join(er.names)}. Tested {fi} out of {len(files[m])} files")
                X, y = get_X_y(file)

                results[file.name] = er.predict(X, y, last_month=last_month, actual_month=actual_month, file_num=fi, update=update)

                results_temp = deepcopy(results[file.name])
                results_temp['month'] = mn
                results_temp['year'] = test_year
                results_temp['day'] = fi
                save_csv(output_day.joinpath(f'{mn}.csv'), results_temp, logger=logger)
            end = perf_counter() - start

            name = get_cls_name(er) if not update else f'{get_cls_name(er)}_update'

            time = {'clf': name, 'time_elapsed': end, 'type': 'test', 'month': actual_month}
            mutex.acquire()
            save_csv(output_path.joinpath('time_elapsed.csv'), time, logger=LOGGER)
            mutex.release()

            # Parsing results
            ensemble_metrics = show_results(results)
            ensemble_metrics['month'] = mn
            ensemble_metrics['test_year'] = test_year

            save_csv(filename, ensemble_metrics, logger=LOGGER)
            last_month = actual_month

@timer(logger=LOGGER)
def main_classify_rejection(update=False):
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

    mutex = Lock()
    Thread(target=_test_rejection, args=(CLASSIFIERS, CLASSIFIER_THRESHOLDS, files, FROM, TO, OUTPUT, LOGGER, False, mutex)).start()
    Thread(target=_test_rejection, args=(CLASSIFIERS, CLASSIFIER_THRESHOLDS, files, FROM, TO, OUTPUT, LOGGER, True, mutex)).start()

def _test_batch(classifier, files, output: Path, month, _from, _to):
    results = dict()
    clf_name = get_cls_name(classifier)
    csv_name = f'{clf_name}_{_from}_{_to}.csv'
    output_day = output.joinpath(f'per_day_{clf_name}')

    if not output_day.exists():
        output_day.mkdir()

    for day, file in enumerate(files):
        if not isinstance(classifier, StreamVotingClassifier):
            results[file.name] = get_metrics(classifier, file)
        else:
            X, y = get_X_y(file)
            res = classifier.predict(X, y)
            results[file.name] = res
        results_temp = deepcopy(results[file.name])
        results_temp['month'] = month
        results_temp['year'] = _from
        results_temp['day'] = day
        save_csv(output_day.joinpath(f"{month}.csv"), results_temp, logger=LOGGER)

    metrics = show_results(results)
    metrics['month'] = month
    metrics['year'] = _from
    save_csv(output.joinpath(csv_name), metrics, logger=LOGGER)


def main_classify_batch(update=False):
    VIEW = 'MOORE'
    FROM = 2014
    TO = 2015
    OUTPUT = CSV_PATH.joinpath('batch_classifiers_update')
    CLASSIFIERS = [['rf', RandomForestClassifier()], ['gbt', GradientBoostingClassifier()], ['ada', AdaBoostClassifier()]]
    N_DAYS = 7

    if not OUTPUT.exists():
        OUTPUT.mkdir()

    files = {month: [r.choice(get_files(FROM, VIEW, month)['files']) for _ in range(N_DAYS)] for month, _ in MONTHS}

    # Preparing dataset
    X = []
    y = []
    for file in files['01']:
        _X, _y = get_X_y(file)
        X += _X.tolist()
        y += _y.tolist()

    # Removing nested lists
    # X = flatten(X)
    # y = flatten(y)
    # embed()

    # Training classifiers
    LOGGER.info(f"Started training classifiers {list(map(lambda x: get_cls_name(x[1]), CLASSIFIERS))}")
    for i in range(len(CLASSIFIERS)):
        start = perf_counter()
        LOGGER.info(f"Training classifier {get_cls_name(CLASSIFIERS[i][1])}")
        CLASSIFIERS[i][1] = CLASSIFIERS[i][1].fit(X, y)
        end = perf_counter() - start
        times = {'clf': CLASSIFIERS[i][0], 'time_elapsed': end, 'type': 'train', 'month': '01'}
        save_csv(OUTPUT.joinpath("time_elapsed.csv"), times, logger=LOGGER)

        LOGGER.info(f"Saved model {CLASSIFIERS[i][0]}")

    # Training ensemble
    start = perf_counter()
    LOGGER.info(f"Started training ensemble")
    ens = VotingClassifier(estimators=CLASSIFIERS, voting='hard')
    ens = ens.fit(X, y)
    end = perf_counter() - start

    # Saving time results
    times = {'clf': 'Ensemble', 'time_elapsed': end, 'type': 'train', 'month': '01'}
    save_csv(OUTPUT.joinpath("time_elapsed.csv"), times, logger=LOGGER)

    # Function to be used as a thread
    def test_classifiers(classifier: object, files: dict, mutex: Lock, update=False):
        # Testing classifiers
        current_month = '01'
        last_month = current_month

        clf_name = classifier[0]
        clf = classifier[1]

        for month in files:
            _files = files[month]
            
            current_month = month
            if update and current_month != last_month and current_month != '02':
            # if update:
                LOGGER.info(f"Started training {clf_name} with month {last_month}")
                train_files = files[last_month]

                # Turning a single month into a single list. 
                # This avoid the model to be scrapped
                X = []
                y = []
                for file in train_files:
                    _X, _y = get_X_y(file)
                    X += _X.tolist()
                    y += _y.tolist()

                try:
                    start_train = perf_counter()
                    clf = clf.fit(X, y)
                    end_train = perf_counter() - start_train
                    time = {'clf': clf_name, 'time_elapsed': end_train, 'type': 'train', 'month': last_month}
                    mutex.acquire()
                    save_csv(OUTPUT.joinpath('time_elapsed.csv'), time, logger=LOGGER)
                    mutex.release()
                except:
                    from traceback import print_exc
                    from io import StringIO
                    s = StringIO()
                    print_exc(file=s)
                    LOGGER.exception(s.readlines())

            last_month = month

            # Testing
            LOGGER.info(f"Started testing {clf_name} at month {month}")
            start = perf_counter()
            _test_batch(clf, _files, OUTPUT, month, FROM, TO)
            end = perf_counter() - start

            # Saving results
            times = {'clf': clf_name, 'time_elapsed': end, 'type': 'test', 'month': month}
            mutex.acquire()
            save_csv(OUTPUT.joinpath("time_elapsed.csv"), times, logger=LOGGER)
            mutex.release()

        LOGGER.info(f"Finished {clf_name}")

    mutex = Lock()
    for clf in [['VotingClassifier', ens]]:
        Thread(target=test_classifiers, args=(clf, files, mutex), kwargs={'update': True}).start()

if __name__ == '__main__':
    # main_classifiers(update=True)
    # main_ensemble()
    # for clf in MODELS_PATH.glob('*.model'):
    #     main_rejection(clf.name)
    # main_classify_rejection()

    # for file in CSV_PATH.joinpath('pareto').glob('*.csv'):
    #     if 'Classifier' in file.name:
    #         get_rejection_metrics(file)
    
    # for file in CSV_PATH.joinpath('rejection_metrics').glob('*.csv'):
    #     if 'Classifier' in file.name:
    #         compute_pareto(file)

    # plot_results()

    main_classify_batch(update=True)