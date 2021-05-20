from constants import DATASETS, MODELS_PATH
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from time import perf_counter
from functools import wraps
from logging import Logger
import pandas as pd
import pickle
import sys

__all__ = ['get_files', 'get_X_y', 'mean_accuracy', 'std_accuracy', 'clf_predict', 'show_results', 'mean', 'timer']

def print_progress(actual, total, label=None):
    msg = f"{actual}/{total} - {(actual / total * 100):.2f}%\r"
    if label and isinstance(label, str):
        msg = f"{label}\t{msg}"
    sys.stdout.write(msg)
    sys.stdout.flush()

def get_files(year: int, base: str, month: str, ext="csv") -> list:
    """ Get files from a given year and month from a dataset in MAWI laab dataset """
    global DATASETS
    ret = {
        'path': DATASETS[year][base].joinpath(month),
        'files': [f for f in DATASETS[year][base].joinpath(month).glob(f"*.{ext}")]
    }
    ret['files'] = sorted(ret['files'])
    return ret

def get_X_y(csv: Path) -> tuple:
    """ Get and returns (X, y) from a given CSV from MAWI lab dataset """
    unwanted_columns = [
        'MAWILAB_taxonomy', 'MAWILAB_label', 
        'MAWILAB_nbDetectors', 'MAWILAB_distance',
    ]

    df = pd.read_csv(csv)
    df = df.sample(frac=1)
    df = df.drop(unwanted_columns, axis=1)

    if 'VIEGAS' in csv.name:
        df = df.drop(['VIEGAS_numberOfDifferentDestinations_A', 'VIEGAS_numberOfDifferentServices_A'], axis=1)
    elif 'ORUNADA' in csv.name:
        df = df.drop(['ORUNADA_numberOfDifferentDestinations', 'ORUNADA_numberOfDifferentServices'], axis=1)
        
    X = df.drop(['class'], axis=1).to_numpy()
    y = df['class'].to_numpy()
    return X, y

def mean_accuracy(results: dict) -> float:
    """ Computes the average accuracy from results returned from 'clf_predict' function """
    x = [results[i]['accuracy'] for i in results]
    return sum(x) / len(x)

def std_accuracy(results: dict) -> float:
    """ 
    Computes standard deviation from the average accuracy from results returned 
    from 'clf_predict' function
    """
    x = [results[i]['accuracy'] for i in results]
    mean = mean_accuracy(results)
    std = (sum((i - mean) ** 2 for i in x) / (len(x) - 1)) ** 0.5
    return std

def clf_predict(clf: object, files: list, logger) -> dict:
    """ Computes classifier metrics from a CSV file within MAWI lab dataset """
    results = dict()
    for file in files:
        logger.debug(f"Testing file {file} in {clf.__class__.__name__}")
        corrects = 0
        samples = 0
        fn = 0
        fp = 0
        tp = 0
        tn = 0
        X_new, y_new = get_X_y(file)

        if not isinstance(clf, RandomForestClassifier):
            for i in range(len(X_new)):
                pred = clf.predict([X_new[i]])
                if pred is not None:
                    if pred[0] == y_new[i]:
                        corrects += 1
                    # False negative
                    if pred[0] == 0 and y_new[i] == 1:
                        fn += 1
                    # False positive
                    if pred[0] == 1 and y_new[i] == 0:
                        fp += 1
                    # True positive
                    if pred[0] == 1 and y_new[i] == 1:
                        tp += 1
                    # False negative
                    if pred[0] == 0 and y_new[i] == 0:
                        tn += 1
                samples += 1
        else:
            # https://stackoverflow.com/a/46230267
            pred = clf.predict(X_new)
            tn, fp, fn, tp = confusion_matrix(y_new, pred, labels=[0, 1]).ravel()
        
        # FNR: False Negative Rate
        # FPR: False Positive Rateshow_results
        results[file.name] = {
            'corrects': corrects, 
            'samples': samples, 
            'fp': fp,
            'fn': fn,
            'fnr': fn / (fn + tp),
            'fpr': fp / (fp + tn),
            'tp': tp,
            'tn': tn,
            'recall': tp / (tp + fn),
            'precision': tp / (tp + fp),
            'accuracy': round(clf.score(X_new, y_new), 5),
        }
    return results

mean = lambda results, key: sum(results[k][key] for k in results) / len(results)

def show_results(results: dict) -> dict:
    """ Show metrics returned from 'clf_predict' prettier """
    mean_acc = mean_accuracy(results)
    mean_std = std_accuracy(results)
    precision = mean(results, 'precision')
    recall = mean(results, 'recall')
    fpr = mean(results, 'fpr')
    fnr = mean(results, 'fnr')
    fp = mean(results, 'fp')
    fn = mean(results, 'fn')
    tp = mean(results, 'tp')
    tn = mean(results, 'tn')

    print("-*" * 40)
    print(f"Mean accuracy: {mean_acc} \t Mean Accuracy Std: {mean_std}")
    print(f"FN: {fn} \t FP: {fp}")
    print(f"TP: {tp} \t TN: {tn}")
    print(f"FPR: {fpr} \t FNR: {fnr}")
    print(f"Precision: {precision} \t Recall: {recall}")
    
    ret = {
        'precision': precision, 'recall': recall, 
        'fpr': fpr, 'fnr': fnr, 
        'fp': fp, 'tn': tn, 
        'fn': fn, 'tp': tp, 
        'mean_accuracy': mean_acc, 'mean_std': mean_std
    }
    
    return ret


def timer(logger=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = perf_counter()
            ret = func(*args, **kwargs)
            end = perf_counter()
            msg = f"Function {func.__name__} took {end - start:.4f}s to complete with arguments {args} and kwargs {kwargs}"

            if logger and isinstance(logger, Logger):
                logger.info(msg)
            else:
                print(msg)
            return ret
        return wrapper
    return decorator

def save_model(model: object, model_name: str) -> None:
    _model = pickle.dumps(model)

    with open(MODELS_PATH.joinpath(model_name + ".model"), 'wb') as fd:
        fd.write(_model)

def load_model(model_name: str) -> object:
    with open(MODELS_PATH.joinpath(model_name), 'rb') as fd:
        return pickle.loads(fd.read())

def ensemble_predict(classifiers: list, X: list):
    predictions = [clf.predict(X)[0] for _, clf in classifiers]
    votes_by_class = ((predictions[i], predictions.count(obj)) for i, obj in enumerate(predictions))
    votes_by_class = sorted(votes_by_class, key=lambda x: x[1])
    return {'class': votes_by_class[-1][0], 'votes': votes_by_class[-1][1]}

def ensemble(classifiers: list, files: list, logger: Logger):
    results = dict()

    for file in files:
        logger.debug(f"Testing file {file} in ensemble")
        corrects = 0
        samples = 0
        fn = 0
        fp = 0
        tp = 0
        tn = 0
        X_new, y_new = get_X_y(file)

        for i in range(len(X_new)):
            pred = ensemble_predict(classifiers, [X_new[i]])
            if pred is not None:
                if pred['class'] == y_new[i]:
                    corrects += 1
                # False negative
                if pred['class'] == 0 and y_new[i] == 1:
                    fn += 1
                # False positive
                if pred['class'] == 1 and y_new[i] == 0:
                    fp += 1
                # True positive
                if pred['class'] == 1 and y_new[i] == 1:
                    tp += 1
                # False negative
                if pred['class'] == 0 and y_new[i] == 0:
                    tn += 1
            samples += 1
            print_progress(samples, len(X_new), label=file.name)
        
        # FNR: False Negative Rate
        # FPR: False Positive Rateshow_results
        results[file.name] = {
            'corrects': corrects, 
            'samples': samples, 
            'fp': fp,
            'fn': fn,
            'fnr': fn / (fn + tp),
            'fpr': fp / (fp + tn),
            'tp': tp,
            'tn': tn,
            'recall': tp / (tp + fn),
            'precision': tp / (tp + fp),
            'accuracy': round(corrects / samples, 5),
        }
    return results