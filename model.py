import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def jenson_shannon_divergence(p, q):
    E_positive = (np.log(2) - F.softplus(- p)).mean()
    E_negative = (- np.log(2) + q + F.softplus(- q)).mean()
    return - (E_positive - E_negative)


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class GlobalPatchDiscriminator(nn.Module):
    """Discriminator model for comparing graph/patch representation pairs"""
    def __init__(self, global_encoder, patch_encoder):
        super().__init__()
        self.global_encoder = global_encoder
        self.patch_encoder = patch_encoder

    def forward(self, x_global, x_patch, batch):
        h_global = self.global_encoder(x_global)
        h_patch = self.patch_encoder(x_patch)
        score_matrix = torch.mm(h_global, h_patch.T)

        # compute real-fake label mask
        mask = torch.zeros_like(score_matrix, dtype=torch.bool, device='cuda')
        mask[batch, torch.arange(len(batch))] = 1

        real_scores = score_matrix[mask]
        fake_scores = score_matrix[~mask]
        loss = jenson_shannon_divergence(real_scores, fake_scores)

        return loss


class GlobalGlobalDiscriminator(nn.Module):
    """Discriminator model for comparing graph/patch representation pairs"""
    def __init__(self, encoder1, encoder2):
        super().__init__()
        self.global_encoder1 = encoder1
        self.global_encoder2 = encoder2

    def forward(self, x1, x2):
        h1 = self.global_encoder1(x1)
        h2 = self.global_encoder2(x2)
        score_matrix = torch.mm(h1, h2.T)

        # compute real-fake label mask
        mask = torch.eye(h1.shape[0], dtype=torch.bool, device='cuda')

        real_scores = score_matrix[mask]
        fake_scores = score_matrix[~mask]
        loss = jenson_shannon_divergence(real_scores, fake_scores)

        return loss
