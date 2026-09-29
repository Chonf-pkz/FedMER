import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, layer_dims, num_classes, dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in layer_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(out_dim))            
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
