import torch
import os

import json
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from utils import move_batch_to_device
from config import Config

from sklearn.metrics import (

    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    accuracy_score,
    matthews_corrcoef,
    consensus_score,
    roc_auc_score,
    recall_score,
    precision_score,
    average_precision_score, confusion_matrix

)
from tqdm import tqdm



# ============================================================
# Test Evaluation
# ============================================================

@torch.no_grad()
def evaluate_test(

        model,

        test_loader,

        device

):

    model.eval()

    ##########################################################

    all_labels = []

    all_probs = []

    all_preds = []

    all_ids = []

    ##########################################################

    for batch in tqdm(

            test_loader,

            desc="Testing"


    ):
        all_ids.extend(batch["id"])
        ######################################################

        batch = move_batch_to_device(

            batch,

            device

        )


        ######################################################

        labels = batch["label"].float().unsqueeze(1)

        ######################################################

        logits, _ = model(batch)

        ######################################################

        probs = torch.sigmoid(logits)

        preds = (probs >= Config.THRESHOLD ).float()

        ######################################################

        all_labels.extend(

            labels.cpu().numpy().flatten()

        )

        all_probs.extend(

            probs.cpu().numpy().flatten()

        )

        all_preds.extend(

            preds.cpu().numpy().flatten()

        )

    ##########################################################

    all_labels = np.array(all_labels)

    all_probs = np.array(all_probs)

    all_preds = np.array(all_preds)

    ##########################################################

    return {
        "ids": all_ids,

        "labels": all_labels,

        "probabilities": all_probs,

        "predictions": all_preds

    }
# Compute Test Metrics
# ============================================================

def compute_test_metrics(results):

    labels = results["labels"]

    probs = results["probabilities"]

    preds = results["predictions"]

    ##########################################################

    tn, fp, fn, tp = confusion_matrix(

        labels,

        preds

    ).ravel()

    ##########################################################

    specificity = tn / (tn + fp + 1e-8)

    ##########################################################

    metrics = {

        "accuracy":
            accuracy_score(labels, preds),

        "precision":
            precision_score(
                labels,
                preds,
                zero_division=0
            ),

        "recall":
            recall_score(
                labels,
                preds,
                zero_division=0
            ),

        "specificity":
            specificity,

        "f1":
            f1_score(
                labels,
                preds,
                zero_division=0
            ),

        "mcc":
            matthews_corrcoef(labels, preds),

        "auc":
            roc_auc_score(labels, probs),

        "auprc":
            average_precision_score(labels, probs),

        "confusion":
            np.array([[tn, fp], [fn, tp]]),

        "n_test":
            len(labels),

        "threshold":
            Config.THRESHOLD,

        ##########################################################
        # Confidence Interval
        ##########################################################

        "confidence_interval": {

            "AUROC": None,

            "AUPRC": None,

            "MCC": None,

            "F1": None

        }

    }
    ##########################################################
    # Bootstrap Confidence Intervals
    ##########################################################

    metrics["confidence_interval"] = bootstrap_confidence_intervals(

        labels,

        probs,

        preds

    )
    return metrics

# ============================================================
# ============================================================
# ROC Curve
# ============================================================

def plot_roc_curve(

        labels,

        probabilities,

        save_dir="results"

):
    """
    Plot ROC curve.

    Parameters
    ----------
    labels : ndarray

    probabilities : ndarray

    save_dir : str
    """

    ##########################################################

    os.makedirs(

        save_dir,

        exist_ok=True

    )

    ##########################################################

    fpr, tpr, _ = roc_curve(

        labels,

        probabilities

    )

    roc_auc = auc(

        fpr,

        tpr

    )

    ##########################################################

    plt.figure(

        figsize=(7, 6)

    )

    ##########################################################

    plt.plot(

        fpr,

        tpr,

        color="darkred",

        linewidth=2.5,

        label=f"AUROC = {roc_auc:.4f}"

    )

    ##########################################################

    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--",

        color="gray",

        linewidth=1.5

    )

    ##########################################################

    plt.xlim(0, 1)

    plt.ylim(0, 1.05)

    ##########################################################

    plt.xlabel(

        "False Positive Rate",

        fontsize=13

    )

    plt.ylabel(

        "True Positive Rate",

        fontsize=13

    )

    plt.title(

        "Receiver Operating Characteristic (ROC)",

        fontsize=14,

        weight="bold"

    )

    ##########################################################

    plt.legend(

        loc="lower right",

        fontsize=12

    )

    ##########################################################

    plt.grid(

        alpha=0.30

    )

    ##########################################################

    plt.tight_layout()

    ##########################################################

    save_path = os.path.join(

        save_dir,

        "roc_curve.jpeg"

    )

    plt.savefig(

        save_path,

        dpi=600,

        bbox_inches="tight",

        format="jpeg"

    )

    plt.close()

    ##########################################################

    print()

    print("ROC curve saved:")

    print(save_path)

# ============================================================
# Precision-Recall Curve
# ============================================================

def plot_pr_curve(

    labels,

    probabilities,

    save_dir="results"

):
    """
    Plot Precision-Recall curve.

    Parameters
    ----------
    labels : ndarray

    probabilities : ndarray

    save_dir : str
    """

    ##########################################################

    os.makedirs(

        save_dir,

        exist_ok=True

    )

    ##########################################################

    precision, recall, _ = precision_recall_curve(

        labels,

        probabilities

    )

    ##########################################################

    ap = average_precision_score(

        labels,

        probabilities

    )

    ##########################################################

    plt.figure(

        figsize=(7, 6)

    )

    ##########################################################

    plt.plot(

        recall,

        precision,

        color="darkblue",

        linewidth=2.5,

        label=f"AUPRC = {ap:.4f}"

    )

    ##########################################################

    plt.xlim(0, 1)

    plt.ylim(0, 1.05)

    ##########################################################

    plt.xlabel(

        "Recall",

        fontsize=13

    )

    plt.ylabel(

        "Precision",

        fontsize=13

    )

    ##########################################################

    plt.title(

        "Precision–Recall Curve",

        fontsize=14,

        weight="bold"

    )

    ##########################################################

    plt.grid(

        alpha=0.30

    )

    ##########################################################

    plt.legend(

        loc="lower left",

        fontsize=12

    )

    ##########################################################

    plt.tight_layout()

    ##########################################################

    save_path = os.path.join(

        save_dir,

        "pr_curve.jpeg"

    )

    plt.savefig(

        save_path,

        dpi=600,

        bbox_inches="tight",

        format="jpeg"

    )

    plt.close()

    ##########################################################

    print()

    print("PR curve saved:")

    print(save_path)


# ============================================================
# Confusion Matrix
# ============================================================

def plot_confusion_matrix(

        labels,

        predictions,

        save_dir="results"

):
    """
    Plot confusion matrix.

    Parameters
    ----------
    labels : ndarray

    predictions : ndarray

    save_dir : str
    """

    ##########################################################

    os.makedirs(

        save_dir,

        exist_ok=True

    )

    ##########################################################

    cm = confusion_matrix(

        labels,

        predictions

    )

    ##########################################################

    class_names = [

        "Non-ACP",

        "ACP"

    ]

    ##########################################################

    fig, ax = plt.subplots(

        figsize=(6, 6)

    )

    ##########################################################

    im = ax.imshow(

        cm,

        cmap="Blues"

    )

    ##########################################################

    ax.set_xticks(

        np.arange(len(class_names))

    )

    ax.set_yticks(

        np.arange(len(class_names))

    )

    ##########################################################

    ax.set_xticklabels(

        class_names,

        fontsize=12

    )

    ax.set_yticklabels(

        class_names,

        fontsize=12

    )

    ##########################################################

    ax.set_xlabel(

        "Predicted Label",

        fontsize=13,

        weight="bold"

    )

    ax.set_ylabel(

        "True Label",

        fontsize=13,

        weight="bold"

    )

    ##########################################################

    ax.set_title(

        "Confusion Matrix",

        fontsize=14,

        weight="bold"

    )

    ##########################################################

    thresh = cm.max() / 2

    ##########################################################

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(

                j,

                i,

                str(cm[i, j]),

                ha="center",

                va="center",

                fontsize=14,

                color="white"

                if cm[i, j] > thresh

                else "black"

            )

    ##########################################################

    fig.colorbar(

        im,

        ax=ax,

        fraction=0.046,

        pad=0.04

    )

    ##########################################################

    plt.tight_layout()

    ##########################################################

    save_path = os.path.join(

        save_dir,

        "confusion_matrix.jpeg"

    )

    plt.savefig(

        save_path,

        dpi=600,

        bbox_inches="tight",

        format="jpeg"

    )

    plt.close()

    ##########################################################

    print()

    print("Confusion Matrix saved:")

    print(save_path)
# ============================================================
# Save Test Predictions
# ============================================================



def save_predictions(

        results,

        ids,

        save_dir="results"

):
    """
    Save test predictions to CSV.

    Parameters
    ----------
    results : dict

    ids : list

    save_dir : str
    """

    ##########################################################

    os.makedirs(

        save_dir,

        exist_ok=True

    )

    ##########################################################

    df = pd.DataFrame(

        {

            "ID": ids,

            "TrueLabel": results["labels"],

            "Prediction": results["predictions"],

            "Probability": results["probabilities"]

        }

    )

    ##########################################################

    save_path = os.path.join(

        save_dir,

        "test_predictions.csv"

    )

    ##########################################################

    df.to_csv(

        save_path,

        index=False

    )

    ##########################################################

    print()

    print("Predictions saved:")

    print(save_path)

    ##########################################################

    return df
# ============================================================
# Load Best Checkpoint
# ============================================================
def load_best_model(

        model,

        device,

        checkpoint_path="./checkpoints/best_model.pt"

):
    """
    Load best saved checkpoint.

    Parameters
    ----------
    model

    device

    checkpoint_path

    Returns
    -------
    model
    """

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(

            checkpoint_path

        )

    ##########################################################

    model.load_state_dict(

        torch.load(

            checkpoint_path,

            map_location=device

        )

    )

    model.to(device)

    model.eval()

    ##########################################################

    print()

    print("=" * 60)

    print("Best checkpoint loaded successfully.")

    print(checkpoint_path)

    print("=" * 60)

    ##########################################################

    return model
# ============================================================
# Save Metrics (JSON)
# ============================================================

def save_metrics_json(

        metrics,

        save_path="./results/test_metrics.json"

):
    """
    Save evaluation metrics to JSON file.
    """

    ##########################################################

    os.makedirs(

        os.path.dirname(save_path),

        exist_ok=True

    )

    ##########################################################

    output = {}

    for key, value in metrics.items():

        ######################################################

        if isinstance(value, np.ndarray):

            output[key] = value.tolist()

        ######################################################

        elif isinstance(value, np.floating):

            output[key] = float(value)

        ######################################################

        elif isinstance(value, np.integer):

            output[key] = int(value)

        ######################################################

        else:

            output[key] = value

    ##########################################################

    with open(

            save_path,

            "w",

            encoding="utf-8"

    ) as f:

        json.dump(

            output,

            f,

            indent=4

        )

    ##########################################################

    print()

    print("Metrics saved:")

    print(save_path)
# ============================================================
# Print Final Test Results
# ============================================================

def print_test_results(

        metrics

):
    """
    Print test evaluation metrics.
    """

    print()

    print("=" * 60)

    print("Final Test Results")

    print("=" * 60)

    print()

    print(f"Accuracy     : {metrics['accuracy']:.4f}")

    print(f"Precision    : {metrics['precision']:.4f}")

    print(f"Recall       : {metrics['recall']:.4f}")

    print(f"Specificity  : {metrics['specificity']:.4f}")

    print(f"F1-score     : {metrics['f1']:.4f}")

    print(f"MCC          : {metrics['mcc']:.4f}")

    print(f"AUROC        : {metrics['auc']:.4f}")

    print(f"AUPRC        : {metrics['auprc']:.4f}")

    print()

    #print("Confusion Matrix")

    #print(metrics["confusion"])

    print()

    print("=" * 60)
    ci = metrics["confidence_interval"]

    print()
    print("95% Bootstrap Confidence Intervals")
    print("-" * 40)

    print(
        f"AUROC : [{ci['AUROC'][0]:.4f}, {ci['AUROC'][1]:.4f}]"
    )

    print(
        f"AUPRC : [{ci['AUPRC'][0]:.4f}, {ci['AUPRC'][1]:.4f}]"
    )

    print(
        f"MCC   : [{ci['MCC'][0]:.4f}, {ci['MCC'][1]:.4f}]"
    )

    print(
        f"F1    : [{ci['F1'][0]:.4f}, {ci['F1'][1]:.4f}]"
    )


def generate_all_figures(

        results,

        metrics,

        output_dir="./results"

):
    """
    Generate all evaluation figures and files.

    Parameters
    ----------
    results : dict
        Output of evaluate_test()

    metrics : dict
        Output of compute_metrics()

    output_dir : str
        Directory to save figures/files.
    """

    os.makedirs(output_dir, exist_ok=True)

    print()
    print("=" * 60)
    print("Generating Evaluation Figures ...")
    print("=" * 60)


    plot_pr_curve(

        results["labels"],

        results["probabilities"],

        save_dir = output_dir

    )
    plot_roc_curve(

        results["labels"],

        results["probabilities"],

        save_dir=output_dir

    )

    ##########################################################
    # Confusion Matrix
    ##########################################################

    plot_confusion_matrix(

        results["labels"],

        results["predictions"],

        save_dir = output_dir

    )
    ##########################################################
    # Probability Distribution
    ##########################################################

    plot_probability_distribution(

        results["labels"],

        results["probabilities"],

        save_dir = output_dir


    )

    ##########################################################
    # Prediction CSV
    ##########################################################

    save_predictions(

        results,

        os.path.join(output_dir, "test_predictions.csv")

    )

    ##########################################################
    # Metrics JSON
    ##########################################################
    print("Saving metrics to:", os.path.join(output_dir, "test_metrics.json"))
    print("Is directory:",
          os.path.isdir(os.path.join(output_dir, "test_metrics.json")))
    save_metrics_json(

        metrics,

        os.path.join(output_dir, "test_metrics.json")

    )

    ##########################################################
    # Print Results
    ##########################################################

    print_test_results(

        metrics

    )

    print()
    print("=" * 60)
    print("Evaluation Finished")
    print("=" * 60)

    plot_roc_pr_figure(

        results["labels"],

        results["probabilities"],

        save_dir=output_dir

    )

def plot_probability_distribution(

        labels,

        probabilities,

        save_dir="results"

):

    """
    Kernel Density Estimation (KDE)
    of predicted probabilities.
    """

    os.makedirs(

        save_dir,

        exist_ok=True

    )

    ##########################################################

    labels = np.asarray(labels)

    probabilities = np.asarray(probabilities)

    ##########################################################

    positive = probabilities[labels == 1]

    negative = probabilities[labels == 0]

    ##########################################################

    x = np.linspace(

        0,

        1,

        500

    )

    ##########################################################

    kde_positive = gaussian_kde(

        positive

    )

    kde_negative = gaussian_kde(

        negative

    )

    ##########################################################

    plt.figure(

        figsize=(7,6)

    )

    ##########################################################

    plt.fill_between(

        x,

        kde_negative(x),

        alpha=0.35,

        color="steelblue"

    )

    plt.plot(

        x,

        kde_negative(x),

        linewidth=2.5,

        color="steelblue",

        label="Non-ACP"

    )

    ##########################################################

    plt.fill_between(

        x,

        kde_positive(x),

        alpha=0.35,

        color="darkorange"

    )

    plt.plot(

        x,

        kde_positive(x),

        linewidth=2.5,

        color="darkorange",

        label="ACP"

    )

    ##########################################################

    plt.xlim(

        0,

        1

    )

    ##########################################################

    plt.xlabel(

        "Predicted Probability",

        fontsize=13,

        weight="bold"

    )

    plt.ylabel(

        "Density",

        fontsize=13,

        weight="bold"

    )

    ##########################################################

    plt.title(

        "Kernel Density Estimation of Prediction Scores",

        fontsize=14,

        weight="bold"

    )

    ##########################################################

    plt.grid(

        alpha=0.30

    )

    ##########################################################

    plt.legend(

        fontsize=12

    )

    ##########################################################

    plt.tight_layout()

    ##########################################################

    save_path = os.path.join(

        save_dir,

        "probability_distribution.jpeg"

    )

    plt.savefig(

        save_path,

        dpi=600,

        bbox_inches="tight",

        format="jpeg"

    )

    plt.close()

    ##########################################################

    print()

    print("Probability KDE saved:")

    print(save_path)
def plot_roc_pr_figure(

        labels,

        probabilities,

        save_dir="results"

):
    """
    Combined ROC + PR Figure
    suitable for publication.
    """

    os.makedirs(

        save_dir,

        exist_ok=True

    )

    ##########################################################

    fpr, tpr, _ = roc_curve(

        labels,

        probabilities

    )

    roc_auc = auc(

        fpr,

        tpr

    )

    ##########################################################

    precision, recall, _ = precision_recall_curve(

        labels,

        probabilities

    )

    ap = average_precision_score(

        labels,

        probabilities

    )

    ##########################################################

    fig, axes = plt.subplots(

        1,

        2,

        figsize=(12,5)

    )

    ##########################################################
    # (a) ROC
    ##########################################################

    axes[0].plot(

        fpr,

        tpr,

        color="darkred",

        linewidth=2.5,

        label=f"AUROC = {roc_auc:.4f}"

    )

    axes[0].plot(

        [0,1],

        [0,1],

        "--",

        color="gray"

    )

    axes[0].set_xlim(0,1)

    axes[0].set_ylim(0,1.05)

    axes[0].set_xlabel(

        "False Positive Rate",

        fontsize=12

    )

    axes[0].set_ylabel(

        "True Positive Rate",

        fontsize=12

    )

    axes[0].set_title(

        "(a) ROC Curve",

        fontsize=13,

        weight="bold"

    )

    axes[0].grid(alpha=0.30)

    axes[0].legend()

    ##########################################################
    # (b) PR
    ##########################################################

    axes[1].plot(

        recall,

        precision,

        color="darkblue",

        linewidth=2.5,

        label=f"AUPRC = {ap:.4f}"

    )

    axes[1].set_xlim(0,1)

    axes[1].set_ylim(0,1.05)

    axes[1].set_xlabel(

        "Recall",

        fontsize=12

    )

    axes[1].set_ylabel(

        "Precision",

        fontsize=12

    )

    axes[1].set_title(

        "(b) Precision–Recall Curve",

        fontsize=13,

        weight="bold"

    )

    axes[1].grid(alpha=0.30)

    axes[1].legend()

    ##########################################################

    plt.tight_layout()

    ##########################################################

    save_path = os.path.join(

        save_dir,

        "roc_pr_figure.jpeg"

    )

    plt.savefig(

        save_path,

        dpi=600,

        bbox_inches="tight",

        format="jpeg"

    )

    plt.close()

    ##########################################################

    print()

    print("Combined ROC+PR figure saved:")

    print(save_path)
def bootstrap_confidence_intervals(

        labels,

        probabilities,

        predictions,

        n_bootstrap=1000,

        random_state=42

):
    """
    Bootstrap confidence intervals
    for AUROC, AUPRC, MCC and F1.
    """

    rng = np.random.RandomState(

        random_state

    )

    ##########################################################

    auc_scores = []

    auprc_scores = []

    mcc_scores = []

    f1_scores = []

    ##########################################################

    labels = np.asarray(labels)

    probabilities = np.asarray(probabilities)

    predictions = np.asarray(predictions)

    ##########################################################

    N = len(labels)

    ##########################################################

    for _ in range(n_bootstrap):

        idx = rng.randint(

            0,

            N,

            N

        )

        y = labels[idx]

        p = probabilities[idx]

        pred = predictions[idx]

        ##################################################

        if len(np.unique(y)) < 2:

            continue

        ##################################################

        auc_scores.append(

            roc_auc_score(

                y,

                p

            )

        )

        ##################################################

        auprc_scores.append(

            average_precision_score(

                y,

                p

            )

        )

        ##################################################

        mcc_scores.append(

            matthews_corrcoef(

                y,

                pred

            )

        )

        ##################################################

        f1_scores.append(

            f1_score(

                y,

                pred

            )

        )

    ##########################################################

    def CI(values):

        return (

            np.percentile(values, 2.5),

            np.percentile(values, 97.5)

        )

    ##########################################################

    return {

        "AUROC": CI(auc_scores),

        "AUPRC": CI(auprc_scores),

        "MCC": CI(mcc_scores),

        "F1": CI(f1_scores)

    }



