"""
==========================================
metrics.py

KAGB Framework

Evaluation Metrics

==========================================
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


##############################################################
# Accuracy
##############################################################

def accuracy(y_true, y_pred):

    return accuracy_score(y_true, y_pred)


##############################################################
# Precision
##############################################################

def precision(y_true, y_pred):

    return precision_score(
        y_true,
        y_pred,
        zero_division=0
    )


##############################################################
# Recall (Sensitivity)
##############################################################

def sensitivity(y_true, y_pred):

    return recall_score(
        y_true,
        y_pred,
        zero_division=0
    )


##############################################################
# Specificity
##############################################################

def specificity(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    return tn / (tn + fp + 1e-12)


##############################################################
# F1 Score
##############################################################

def f1(y_true, y_pred):

    return f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


##############################################################
# Matthews Correlation Coefficient
##############################################################

def mcc(y_true, y_pred):

    return matthews_corrcoef(
        y_true,
        y_pred
    )


##############################################################
# ROC-AUC
##############################################################

def roc_auc(y_true, y_prob):

    return roc_auc_score(
        y_true,
        y_prob
    )


##############################################################
# PR-AUC
##############################################################

def pr_auc(y_true, y_prob):

    return average_precision_score(
        y_true,
        y_prob
    )


##############################################################
# Complete Evaluation
##############################################################

def compute_metrics(

    y_true,

    y_pred,

    y_prob

):

    metrics = {

        "ACC": accuracy(
            y_true,
            y_pred
        ),

        "Precision": precision(
            y_true,
            y_pred
        ),

        "Sensitivity": sensitivity(
            y_true,
            y_pred
        ),

        "Specificity": specificity(
            y_true,
            y_pred
        ),

        "F1": f1(
            y_true,
            y_pred
        ),

        "MCC": mcc(
            y_true,
            y_pred
        ),

        "ROC_AUC": roc_auc(
            y_true,
            y_prob
        ),

        "PR_AUC": pr_auc(
            y_true,
            y_prob
        )

    }

    return metrics


##############################################################
# Pretty Print
##############################################################

def print_metrics(metrics):

    print("=" * 45)

    print("Evaluation Metrics")

    print("=" * 45)

    for key, value in metrics.items():

        print(f"{key:<15}: {value:.4f}")

    print("=" * 45)