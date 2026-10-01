"""
=========================================================
train_v2.py

SPHR Version 2

=========================================================
"""

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from tqdm import tqdm
import evaluation

############################################################
# Project
############################################################

from config import Config

from EmbeddingDataset import EmbeddingDataset
from contrastive_loss import SupervisedContrastiveLoss

from dataloader import create_dataloader

from model import HCMCAModelV2
from utils import move_batch_to_device

############################################################
# Random Seed
############################################################

seed = 42

random.seed(seed)

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed_all(seed)
############################################################
# Device
############################################################

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

print()

print("=" * 60)

print(f"Device : {device}")

print("=" * 60)

############################################################

############################################################


class BinaryFocalLoss(nn.Module):

    def __init__(
            self,
            alpha=0.5,
            gamma=2.0,
            reduction="mean"
    ):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
            self,
            logits,
            targets
    ):

        targets = targets.float().view_as(logits)

        bce = F.binary_cross_entropy_with_logits(

            logits,

            targets,

            reduction="none"

        )

        pt = torch.exp(-bce)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * (1 - pt) ** self.gamma * bce

        if self.reduction == "mean":

            return loss.mean()

        elif self.reduction == "sum":

            return loss.sum()

        return loss

# ============================================================
# Train One Epoch
# ============================================================
############################################################
# Move batch to device
############################################################

def train_one_epoch(

        model,

        train_loader,

        optimizer,

        criterion,

        contrastive_loss,

        device

):

    model.train()

    total_loss = 0.0

    correct = 0

    total = 0

    progress = tqdm(train_loader)

    for batch_idx, batch in enumerate(progress):

        ########################################################
        # Move tensors to device
        ########################################################

        batch = move_batch_to_device(batch, device)
        labels = batch["label"].float().unsqueeze(1)

        ########################################################
        # Forward
        ########################################################

        logits = model(batch)

        ########################################################
        # Loss
        ########################################################

        logits, embedding = model(batch)

        loss_cls = criterion(logits, labels)

        loss = loss_cls + Config.CONTRASTIVE_WEIGHT * contrastive_loss(
            embedding,
            labels
        )

        ########################################################
        # Backward
        ########################################################

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        ########################################################
        # Statistics
        ########################################################

        total_loss += loss.item()

        probs = torch.sigmoid(logits)

        preds = (probs > Config.THRESHOLD).float()

        correct += (preds == labels).sum().item()

        total += labels.size(0)

        progress.set_postfix(

            loss=f"{loss.item():.4f}",

            acc=f"{100.0*correct/total:.2f}"

        )

        tqdm.write(

            f"Batch {batch_idx+1}/{len(train_loader)}  "

            f"Loss={loss.item():.4f}  "

            f"Acc={100.0*correct/total:.2f}%"

        )

    ########################################################
    # Scheduler
    ########################################################

    ########################################################

    return (

        total_loss / len(train_loader),

        correct / total

    )
# ============================================================
# Validation
# ============================================================

@torch.no_grad()

def validate(

        model,

        val_loader,

        criterion,

        device

):

    model.eval()

    total_loss = 0.0

    all_labels = []

    all_probs = []

    ##########################################################

    for batch in tqdm(val_loader):

        ######################################################
        # Move to device
        ######################################################

        batch = move_batch_to_device(batch, device)

        labels = batch["label"].float().unsqueeze(1)

        ######################################################
        # Forward
        ######################################################

        logits, embedding = model(batch)

        loss = criterion(

            logits,

            labels

        )

        total_loss += loss.item()

        ######################################################
        # Prediction
        ######################################################

        probs = torch.sigmoid(logits)

        all_probs.extend(

            probs.cpu().numpy().flatten()

        )

        all_labels.extend(

            labels.cpu().numpy().flatten()

        )

    ##########################################################
    # Metrics
    ##########################################################

    all_labels = np.array(all_labels)

    all_probs = np.array(all_probs)

    preds = (all_probs > Config.THRESHOLD).astype(int)

    metrics = {}

    metrics["loss"] = total_loss / len(val_loader)

    metrics["accuracy"] = accuracy_score(

        all_labels,

        preds

    )

    metrics["precision"] = precision_score(

        all_labels,

        preds,

        zero_division=0

    )

    metrics["recall"] = recall_score(

        all_labels,

        preds,

        zero_division=0

    )
    ##########################################################
    # Specificity
    ##########################################################

    tn, fp, fn, tp = confusion_matrix(

        all_labels,

        preds

    ).ravel()

    metrics["specificity"] = tn / (tn + fp + 1e-8)

    ##########################################################
    # AUPRC
    ##########################################################

    metrics["auprc"] = average_precision_score(

        all_labels,

        all_probs

    )

    metrics["f1"] = f1_score(

        all_labels,

        preds,

        zero_division=0

    )

    ##########################################################
    # AUC
    ##########################################################

    if len(np.unique(all_labels)) > 1:

        metrics["auc"] = roc_auc_score(

            all_labels,

            all_probs

        )

    else:

        metrics["auc"] = float("nan")

    ##########################################################

    metrics["mcc"] = matthews_corrcoef(

        all_labels,

        preds

    )

    metrics["confusion"] = confusion_matrix(

        all_labels,

        preds

    )

    ##########################################################

    return metrics
# ============================================================
# Test Evaluation
# ============================================================

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
# ============================================================
# Training Loop
# ============================================================

def train(

        model,

        train_loader,

        val_loader

):
    # Loss
    ############################################################

    criterion = nn.BCEWithLogitsLoss()
    # criterion = BinaryFocalLoss(
    #     alpha=0.50,
    #     gamma=2
    # )
    contrastive_loss = SupervisedContrastiveLoss(
        temperature=Config.TEMPERATURE
    )

    ############################################################
    # Optimizer
    ############################################################

    optimizer = optim.AdamW(

        model.parameters(),

        lr=Config.LEARNING_RATE,

        weight_decay=Config.WEIGHT_DECAY

    )

    ############################################################
    # Scheduler
    ############################################################

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=Config.LR_FACTOR,

        patience=Config.LR_PATIENCE,

        threshold=Config.MIN_DELTA,

        min_lr=1e-6

    )

    print()

    print("=" * 60)

    print("Entering train() ...")

    print("=" * 60)

    print("=" * 60)
    print(f"Ablation Mode : {Config.ABLATION_MODE}")
    print("=" * 60)

    ##########################################################
    patience = 0
    ##############################################
    # Best Metrics
    ##############################################

    best_metrics = {

        "auc": -1,

        "accuracy": 0.0,

        "precision": 0.0,

        "recall": 0.0,

        "specificity": 0.0,

        "f1": 0.0,

        "mcc": -1,

        "loss": float("inf")

    }

    ##########################################################

    for epoch in range(Config.EPOCHS):

        ###############################################
        # Train
        ###############################################

        train_loss, train_acc = train_one_epoch(

            model,

            train_loader,

            optimizer,

            criterion,

            contrastive_loss,

            device

        )

        ###############################################
        # Validation
        ###############################################

        metrics = validate(

            model,

            val_loader,

            criterion,

            device

        )
        ###############################################
        # Print
        ###############################################

        print()

        print("=" * 60)

        print(f"Epoch {epoch+1}/{Config.EPOCHS}")

        print("=" * 60)

        print(f"Learning Rate : {optimizer.param_groups[0]['lr']:.2e}")

        print()

        print(f"Train Loss : {train_loss:.4f}")

        print(f"Train Acc  : {train_acc:.4f}")

        print("--------------------------------")

        print(f"Val Loss   : {metrics['loss']:.4f}")

        print(f"Accuracy   : {metrics['accuracy']:.4f}")

        print(f"Precision  : {metrics['precision']:.4f}")

        print(f"Recall     : {metrics['recall']:.4f}")

        print(f"F1-score   : {metrics['f1']:.4f}")

        print(f"AUC        : {metrics['auc']:.4f}")

        print(f"MCC        : {metrics['mcc']:.4f}")

        print()

        #print("Confusion Matrix")

        #print(metrics["confusion"])

        scheduler.step(metrics["auc"])

        ###############################################
        # Save Best
        ###############################################

        if metrics["mcc"] > best_metrics["mcc"] + Config.MIN_DELTA:


            best_metrics = metrics.copy()

            patience = 0

            torch.save(

                model.state_dict(),

                Config.CHECKPOINT_PATH

            )
            print("Saving checkpoint to:", Config.CHECKPOINT_PATH)
            print()

            print("Best model saved.")

        else:

            patience += 1

        ###############################################
        # Early Stopping
        ###############################################

        if patience >= Config.EARLY_STOPPING and epoch > Config.MIN_EPOCH:

            print()

            print("=" * 60)

            print("Early Stopping Activated")

            print("=" * 60)

            break

    ##########################################################

    print()

    print("=" * 60)

    print("Training Finished")

    print("=" * 60)

    print()
    ##########################################################
    # Best Validation Metrics
    ##########################################################

    print()

    print("=" * 60)

    print("Best Validation Metrics")

    print("=" * 60)

    for k, v in best_metrics.items():

        if isinstance(v, float):

            print(f"{k:12s}: {v:.4f}")

        else:

            print(f"{k:12s}: {v}")

# ============================================================
# Load Best Model
# ============================================================

def load_best_model(model, device):
    """
    Load the best checkpoint saved during training.

    Parameters
    ----------
    model : nn.Module

    device : torch.device

    Returns
    -------
    model
    """

    checkpoint_path = os.path.join(

        Config.CHECKPOINT_DIR,

        "best_model_v2.pt"

    )

    ##########################################################

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(

            f"Checkpoint not found:\n{checkpoint_path}"

        )

    ##########################################################

    state_dict = torch.load(

        checkpoint_path,

        map_location=device

    )

    ##########################################################

    model.load_state_dict(

        torch.load(

            Config.CHECKPOINT_PATH

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

    print()

    return model

############################################################
# Main
############################################################

if __name__ == "__main__":

    print("=" * 60)
    print("SPHR Version 2")
    print("Training Using Precomputed Embeddings")
    print("=" * 60)

    ##########################################################
    # Dataset
    ##########################################################

    train_dataset = EmbeddingDataset(
        Config.TRAIN_EMBEDDINGS
    )

    val_dataset = EmbeddingDataset(
        Config.VAL_EMBEDDINGS
    )
    ##########################################################
    # Test Dataset
    ##########################################################

    test_dataset = EmbeddingDataset(

        Config.TEST_EMBEDDINGS

    )
    ##########################################################
    # DataLoader
    ##########################################################

    train_loader = create_dataloader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    ##########################################################
    # Test Loader
    ##########################################################

    test_loader = create_dataloader(

        test_dataset,

        batch_size=Config.BATCH_SIZE,

        shuffle=False

    )

    ##########################################################
    # Information
    ##########################################################

    print()

    print(f"Train Samples      : {len(train_dataset)}")

    print(f"Validation Samples : {len(val_dataset)}")

    print(f"Test Samples       : {len(test_dataset)}")

    print()

    ##########################################################
    # Build Model
    ##########################################################

    model = HCMCAModelV2().to(device)

    ##########################################################
    # Train
    ##########################################################

    train(

        model,

        train_loader,

        val_loader

    )

    ##########################################################
    # Load Best Checkpoint
    ##########################################################

    model = evaluation.load_best_model(

        model,

        device

    )

    ##########################################################
    # Test
    ##########################################################

    results = evaluation.evaluate_test(

        model,

        test_loader,

        device

    )

    metrics = evaluation.compute_test_metrics(

        results

    )
    evaluation.generate_all_figures(

        results,

        metrics,
        output_dir="./results"

    )


