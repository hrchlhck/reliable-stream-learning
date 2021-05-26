from constants import CSV_PATH, DATASETS, MODELS_PATH
from pathlib import Path
from time import perf_counter
from functools import wraps
from logging import Logger
from matplotlib import pyplot as plt
from IPython import embed
import pandas as pd
import pickle
import sys

__all__ = ['get_files', 'get_X_y', 'mean_accuracy', 'std_accuracy', 'clf_predict', 'show_results', 'mean', 'timer', 'get_cls_name']

get_cls_name = lambda obj: obj.__class__.__name__

def get_metrics(classifier, file):
    results = dict()

    corrects = 0
    samples = 0
    fn = 0
    fp = 0
    tp = 0
    tn = 0
    X_new, y_new = get_X_y(file)

    for i in range(len(X_new)):
        pred = classifier.predict([X_new[i]])
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
        print_progress(samples, len(X_new), label=file.name)
    
    # FNR: False Negative Rate
    # FPR: False Positive Rate
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
        results.update(get_metrics(clf, file))
    return results

mean = lambda results, key: sum(results[k][key] for k in results) / len(results)

def show_results(results: dict) -> dict:
    """ Show metrics returned from 'clf_predict' prettier """
    ret = dict()
    for file in results:
        for key in results[file]:
            ret[key] = round(mean(results, key), 4)
    
    print("-*" * 40)
    for key in ret:
        print(f"{key.upper()}: {ret[key]}")
    print("-*" * 40)
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

def get_rejection_metrics(csv: Path):
    filename = Path(csv)
    path = CSV_PATH.joinpath('rejection_metrics')

    if not path.exists():
        path.mkdir()

    start = 50
    total = 100
    step = 2
    df = pd.read_csv(csv)
    output = pd.DataFrame(columns=["lim_attack", "lim_normal", "reject_rate", "error_rate", "fpr", "fnr"])

    for i in range(start, total + 1, step):
        for j in range(start, total + 1, step):
            lim_attack = i / total
            lim_normal = j / total
            
            df_classified_normal = df[(df['predicted_class'] == 0) & (df['normal_confidence'] >= lim_normal)]
            df_classified_attack = df[(df['predicted_class'] == 1) & (df['attack_confidence'] >= lim_attack)]

            reject_normal = df[(df['predicted_class'] == 0) & (df['normal_confidence'] < lim_normal)]
            reject_attack = df[(df['predicted_class'] == 1) & (df['attack_confidence'] < lim_attack)]

            reject_rate = (len(reject_normal) + len(reject_attack)) / len(df)

            tn = df_classified_normal[df_classified_normal['true_class'] == 0]
            tp = df_classified_attack[df_classified_attack['true_class'] == 1]
            fp = df_classified_attack[df_classified_attack['true_class'] == 0]
            fn = df_classified_normal[df_classified_normal['true_class'] == 1]

            fpr = len(fp) / (len(fp) + len(tn))
            fnr = len(fn) / (len(fn) + len(tp))

            error_rate = (fpr + fnr) / 2

            output.loc[len(output)] = [lim_attack, lim_normal, reject_rate, error_rate, fpr, fnr]
    output.to_csv(path.joinpath(filename.name), index=False)

def compute_pareto(csv: Path, plot=False):
    df = pd.read_csv(csv)

    if not CSV_PATH.joinpath('pareto_computed').exists():
        CSV_PATH.joinpath('pareto_computed').mkdir()

    # Create dataframe with columns that matter
    df = df[['reject_rate', 'error_rate', "lim_attack", "lim_normal"]].copy()
    df['dominated'] = False

    for i in range(len(df)):
        if df.loc[i, 'dominated'] == True:
            continue

        for j in range(len(df)):
            reject_i = df.loc[i, 'reject_rate']
            error_i = df.loc[i, 'error_rate']

            reject_j = df.loc[j, 'reject_rate']
            error_j = df.loc[j, 'error_rate']

            if reject_i < reject_j and error_i < error_j:
                df.loc[j, 'dominated'] = True
    
    df_filtered = df[df['dominated'] == False]
    df_filtered = df_filtered.sort_values('error_rate')
    df_filtered.to_csv(f'results/pareto_computed/{csv.name[:-4]}.csv', index=False)

    if plot:
        plt.figure(figsize=(5, 5))
        plt.scatter(df_filtered['reject_rate'], df_filtered['error_rate'], marker='s', color='b', alpha=0.3)
        plt.ylabel('Error rate')
        plt.xlabel('Reject rate')
        plt.savefig(f'images/pareto_{csv.name[:-4]}.png', dpi=210)

def predict_rejection(classifiers: object, rejection_table: dict, files: list, logger: Logger):
    clf_names = [get_cls_name(clf) for clf in classifiers]
    results = dict()

    for count, file in enumerate(files):
        logger.info(f"Testing file {file.name} for {clf_names}. Tested {count} out of {len(files)} files")
        X, y = get_X_y(file)
        rejected = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(len(X)):
            attack_vote = 0
            normal_vote = 0
            predictions = [clf.predict([X[i]]) for clf in classifiers]
            prediction_probabilities = [clf.predict_proba([X[i]]) for clf in classifiers]

            for j in range(len(predictions)):
                pred = predictions[j]
                pred_proba = prediction_probabilities[j]
                
                # If the classifier predicted an attack
                if pred[0] == 1:
                    if pred_proba[0][1] >= rejection_table[clf_names[j]]["attack"]:
                    # if pred_proba[0][1] >= 0.5: 
                        attack_vote += 1
                else:
                    if pred_proba[0][0] >= rejection_table[clf_names[j]]["normal"]:
                    # if pred_proba[0][0] >= 0.99:
                        normal_vote += 1
            
            if attack_vote + normal_vote != 3:
                rejected += 1
            else:
                if normal_vote > attack_vote:
                    if y[i] == 0:
                        tn += 1
                    else:
                        fn += 1
                else:
                    if y[i] == 1:
                        tp += 1
                    else:
                        fp += 1
        results[file.name] = {
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "tn": tn,
            "fnr": fn / (fn + tp),
            "fpr": fp / (fp + tn),
            "rejection_vote": rejected,
            "rejection_rate": rejected / len(X)
        }
        logger.info(f"Tested file {file.name} for {clf_names}. Metrics: {results[file.name]}")
    return results
        
def get_operation_point(classifiers: list, key: str, perc: float):
    """ Get best operation point in non-dominated points """
    ret = dict()
    files = CSV_PATH.joinpath('pareto_computed')

    for i in range(len(classifiers)):
        # Load CSV
        df = pd.read_csv(files / classifiers[i].name)

        # Filtering by a operation point defined by 'perc'
        filtered = df[(df[key] <= perc + 0.01) & (df[key] >= perc)]

        # Getting minimum value for 'key' within the 'perc' value
        filtered = filtered[filtered[key] == filtered[key].min()]

        # Assigning values from each classifier to its' limiar error and limiar attack
        atk = filtered['lim_attack'].tolist()[0]
        norm = filtered['lim_normal'].tolist()[0]
        ret[classifiers[i].name[:-4]] = {"attack": atk, "normal": norm}

    return ret

