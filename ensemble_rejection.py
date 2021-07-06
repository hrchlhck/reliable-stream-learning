from constants import MONTHS
from logging import Logger
from time import perf_counter
from typing import List

from utils import get_cls_name, save_csv
from datetime import datetime
import numpy as np

__all__ = ['EnsembleRejection', 'StreamVotingClassifier']

class EnsembleRejection(object):
    def __init__(self, classifiers: list, thresholds: dict, classes: list, logger=None):
        self.__classifiers = classifiers
        self.__thresholds = thresholds
        self.__classes = classes
        self.__logger = logger
        self.rejections = dict()
    
    @property
    def names(self):
        return [get_cls_name(obj) for obj in self.__classifiers]

    @property
    def classifiers(self):
        return self.__classifiers

    @property
    def classes(self):
        return self.__classes
    
    @property
    def logger(self):
        return self.__logger

    def partial_fit(self, X, y):
        if not isinstance(y, np.ndarray):
            y = np.array(y)
            X = np.array(X)

        if self.__logger and isinstance(self.__logger, Logger):
            self.__logger.debug('Started fitting ensemble')

        for i in range(len(self.classifiers)):
            if self.__logger and isinstance(self.__logger, Logger):
                self.__logger.debug(f'Fitting model {self.names[i]} with {len(X)} instances that probably were rejected')

            for ix in range(len(X)):
                self.__classifiers[i] = self.__classifiers[i].partial_fit([X[ix]], [y[ix]], classes=self.classes)

        if self.__logger and isinstance(self.__logger, Logger):
            self.__logger.debug(f"Finished partial fitting for ensemble with {', '.join(self.names)} models")

    def predict(self, X, y_true, update=False, update_delay=1, **kwargs):
        fp: int = 0
        fn: int = 0
        tp: int = 0
        tn: int = 0
        rejection_count: int = 0
        rejected_instances: List[int] = list()
        actual_month: str = None
        last_month: str = None
        file_num: int = None

        if update == True:
            if not 'actual_month' in kwargs or not 'last_month' in kwargs or not 'file_num' in kwargs:
                raise ValueError('May be missing \'actual_month\', \'last_month\' or \'file_num\' keyword arguments.')

            actual_month = kwargs.get('actual_month')
            last_month = kwargs.get('last_month')
            file_num = kwargs.get('file_num')

            # Training with last month
            if actual_month != last_month and int(last_month) > 0:
                if self.__logger and isinstance(self.__logger, Logger):
                    self.__logger.debug(f'Started updating model with last_month: {MONTHS[last_month]}')
                instances = self.rejections.pop(f'{last_month}-{file_num}')
                X_rej = [x[0] for x in instances]
                y_rej = [y[1] for y in instances]

                start: float = perf_counter()
                self.partial_fit(X_rej, y_rej)
                end: float = perf_counter() - start

                time = {
                    'timestamp': str(datetime.now()), 
                    'clf': get_cls_name(self), 
                    'time_elapsed': end, 
                    'type': 'train', 
                    'month': actual_month, 
                    'last_month': last_month
                }
                
                mutex = kwargs.get('mutex')
                mutex.acquire()
                save_csv(kwargs.get('output'), time, logger=self.__logger)
                mutex.release()
    
        if self.__logger and isinstance(self.__logger, Logger):
            self.__logger.debug("Started evaluation")

        # Evaluation of the ensemble
        for i in range(len(X)):
            attack_vote = 0
            normal_vote = 0
            predictions = [clf.predict([X[i]]) for clf in self.classifiers]
            prediction_probabilities = [clf.predict_proba([X[i]]) for clf in self.classifiers]

            # Voting
            for j in range(len(predictions)):
                pred = predictions[j]
                pred_proba = prediction_probabilities[j]
                
                # If the classifier predicted an attack
                if pred[0] == 1:
                    if pred_proba[0][1] >= self.__thresholds[self.names[j]]["attack"]:
                        attack_vote += 1
                else:
                    if pred_proba[0][0] >= self.__thresholds[self.names[j]]["normal"]:
                        normal_vote += 1
            
            # Rejection
            # # Aceitar apenas se todos aceitarem
            # if attack_vote + normal_vote != 3:
            #     # Increasing rejected instances count and adding into a list of rejections
            #     rejection_count += 1
            #     rejected_instances.append((X[i], y_true[i]))
            # Aceita se a maioria aceitar
            # if attack_vote + normal_vote < 2:
            #     rejection_count += 1
            #     rejected_instances.append((X[i], y_true[i]))
            # Aceita se pelo menos um aceitar 
            if attack_vote + normal_vote == 0:
                rejection_count += 1
                rejected_instances.append((X[i], y_true[i]))
            else:
                # Calculating metrics for evaluation
                if normal_vote > attack_vote:
                    if y_true[i] == 0:
                        tn += 1
                    else:
                        fn += 1
                else:
                    if y_true[i] == 1:
                        tp += 1
                    else:
                        fp += 1
        
        if last_month != None and file_num != None:
            self.rejections[f'{actual_month}-{file_num}'] = rejected_instances

        metrics = {
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "tn": tn,
            "fnr": fn / (fn + tp) if fn > 0 or tp > 0 else 0,
            "fpr": fp / (fp + tn) if tn > 0 or fp > 0 else 0,
            "rejection_vote": rejection_count,
            "rejection_rate": rejection_count / len(X)
        }

        return metrics

class StreamVotingClassifier:
    def __init__(self, *estimators: list, logger=None):
        self.__estimators = list(estimators)
        self.__logger = logger
    
    @property
    def estimators(self):
        return self.__estimators

    @property
    def _logger(self):
        return self.__logger

    @property
    def names(self):
        return [get_cls_name(obj[1]) for obj in self.estimators]
    
    def partial_fit(self, X, y, classes):
        if not isinstance(y, np.ndarray):
            y = np.array(y)
            X = np.array(X)

        for i in range(len(self.estimators)):
            for ix in range(len(X)):
                self.__estimators[i][1] = self.__estimators[i][1].partial_fit([X[ix]], [y[ix]], classes=classes)

        return self

    def predict(self, X, y_true):
        fp = 0
        fn = 0
        tp = 0
        tn = 0

        for i in range(len(X)):
            predictions = [e[1].predict([X[i]]) for e in self.estimators]
            votes_by_class = votes_by_class = ((predictions[i], predictions.count(obj)) for i, obj in enumerate(predictions))
            votes_by_class = sorted(votes_by_class, key=lambda x: x[1])
            final_prediction = {'class': votes_by_class[-1][0], 'votes': votes_by_class[-1][1]}

            if final_prediction['class'] == 1 and y_true[i] == 1:
                tp += 1
            if final_prediction['class'] == 0 and y_true[i] == 0:
                tn += 1
            if final_prediction['class'] == 1 and y_true[i] == 0:
                fp += 1
            if final_prediction['class'] == 0 and y_true[i] == 1:
                fn += 1
        
        results = {
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'tn': tn,
            'fpr': fn / (fn + tp),
            'fnr': fp / (fp + tn),
            'recall': tp / (tp + fn),
            'precision': tp / (tp + fp),
            'accuracy': round((tp + tn) / len(X), 4)
        }
        return results
