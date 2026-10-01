import torch
from torch.utils.data import DataLoader


def create_dataloader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
):

    return DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        num_workers=num_workers,

        pin_memory=torch.cuda.is_available()

    )