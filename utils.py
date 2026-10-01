"""
=========================================================
Utility Functions
=========================================================
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn


# =========================================================
# Random Seed
# =========================================================

def set_seed(seed: int = 42):
    """
    Make experiments reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# Count Parameters
# =========================================================

def count_parameters(model: nn.Module):
    """
    Return number of trainable parameters.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# =========================================================
# Print Model Summary
# =========================================================

def print_model_info(model):

    print("=" * 60)
    print(model)
    print("=" * 60)

    print(f"Trainable Parameters : {count_parameters(model):,}")

    print("=" * 60)


# =========================================================
# Move Batch to Device
# =========================================================

def move_to_device(batch, device):

    if isinstance(batch, dict):

        return {
            k: v.to(device)
            if torch.is_tensor(v)
            else v
            for k, v in batch.items()
        }

    elif isinstance(batch, (list, tuple)):

        return [
            x.to(device)
            if torch.is_tensor(x)
            else x
            for x in batch
        ]

    return batch


# =========================================================
# Save Checkpoint
# =========================================================

def save_checkpoint(model,
                    optimizer,
                    epoch,
                    path):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict()

    }

    torch.save(checkpoint, path)

import torch

def move_batch_to_device(batch, device):
    """
    Move all tensor values of a batch to the target device.

    Non-tensor objects (ids, strings, lists, ...)
    remain unchanged.
    """

    moved = {}

    for key, value in batch.items():

        if torch.is_tensor(value):

            moved[key] = value.to(device)

        else:

            moved[key] = value

    return moved

# =========================================================
# Load Checkpoint
# =========================================================

def load_checkpoint(model,
                    optimizer,
                    path,
                    device):

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    epoch = checkpoint["epoch"]

    return model, optimizer, epoch