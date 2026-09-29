import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCELoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is not None:
            self.register_buffer("weights", torch.tensor(class_weights, dtype=torch.float))
        else:
            self.weights = None

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, weight=self.weights)