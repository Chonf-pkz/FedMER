import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, projection_dim=512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, projection_dim)
        )

    def forward(self, x):
        return self.projection(x)
