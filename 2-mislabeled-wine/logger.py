### MLFLOW helper functions for scikit-learn

import numpy as np
import mlflow
from sklearn.metrics import precision_recall_fscore_support

def log_test(pipe, correct, predicted):
    """Log the results of a train/test run to mlflow"""
    with mlflow.start_run():
        mlflow.set_tag('pipeline', list(map(repr, pipe.named_steps.values())))   
        labels = sorted(set(predicted))
        metrics = precision_recall_fscore_support(correct, predicted, average=None, labels=labels)
        for metric, scores in zip(['precision', 'recall', 'f1'], metrics):
            for cat, score in zip(labels, scores):
                mlflow.log_metric(f'{cat}_{metric}', score)
        metrics = precision_recall_fscore_support(correct, predicted, average='macro')
        for metric, score in zip(['precision', 'recall', 'f1'], metrics):
            mlflow.log_metric(f'{metric}', score)        

def log_search(search):
    """Log f1 scores for a hyperparameter search to mlflow"""
    for params, score in zip(search.cv_results_['params'], search.cv_results_['mean_test_score']):
        with mlflow.start_run():
            mlflow.log_params(params)
            for k,v in params.items():
                if isinstance(v, float):
                    mlflow.log_param('log_'+k, np.log10(v))                
            mlflow.log_metric('mean_test_score', score)