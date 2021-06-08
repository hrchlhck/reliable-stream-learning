from constants import CSV_PATH
from logging import Logger
from pathlib import Path

from IPython.terminal.embed import embed
from utils import get_cls_name, save_csv, timer
import numpy as np

__all__ = ['EnsembleRejection']

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

    def predict(self, X, y_true, **kwargs):
        fp = 0
        fn = 0
        tp = 0
        tn = 0
        rejection_count = 0
        rejected_instances = list()
        actual_month = None
        last_month = None
        file_num = None

        if 'actual_month' in kwargs and 'last_month' in kwargs and 'file_num' in kwargs:
            actual_month = kwargs.get('actual_month')
            last_month = kwargs.get('last_month')
            file_num = kwargs.get('file_num')

            # Training with last month
            if actual_month != last_month:
                print(kwargs)
                instances = self.rejections.pop(f'{last_month}-{file_num}')
                X_rej = [x[0] for x in instances]
                y_rej = [y[1] for y in instances]
                self.partial_fit(X_rej, y_rej)
    
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
