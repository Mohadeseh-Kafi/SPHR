import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.10):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):

        device = embeddings.device

        embeddings = F.normalize(embeddings, dim=1)

        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

        labels = labels.view(-1)

        mask = torch.eq(labels[:, None], labels[None, :]).float().to(device)

        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)

        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask

        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        positive_count = mask.sum(dim=1)

        loss = -(mask * log_prob).sum(dim=1) / (positive_count + 1e-12)

        valid = positive_count > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)

        return loss[valid].mean()
