import torch
import torch.nn as nn
import torch.nn.functional as F

from .GAT_module import GATLayers
from .CBAM import CrossModalFusion
from .nn_utils import SafeBatchNorm1d

class MultiModalGNN(nn.Module):
    def __init__(
        self,
        text_input_dim=768,
        audio_input_dim=768,
        hidden_dim=512,
        num_classes=4,
        dropout=0.3,
        heads=4,
        num_layers=3,
        fusion_head_output_type='min',
        k_text=5,
        k_audio=5,
    ):
        super(MultiModalGNN, self).__init__()

        self.text_projection = nn.Sequential(
            nn.Linear(text_input_dim, hidden_dim),
            SafeBatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.audio_projection = nn.Sequential(
            nn.Linear(audio_input_dim, hidden_dim),
            SafeBatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.cross_fusion = CrossModalFusion(hidden_dim, num_heads=heads, dropout=dropout, fusion_head_output_type=fusion_head_output_type)
        self.gnn = GATLayers(hidden_dim, heads=heads, num_layers=num_layers, dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.k_text = max(0, int(k_text)) if k_text is not None else 0
        self.k_audio = max(0, int(k_audio)) if k_audio is not None else 0
        self.num_classes = num_classes

    def _flatten_modalities(self, text_x, audio_x):
        if text_x.dim() == 3:
            text_x = text_x.mean(dim=1)
        if audio_x.dim() == 3:
            audio_x = audio_x.mean(dim=1)
        if text_x.dim() != 2 or audio_x.dim() != 2:
            raise ValueError("text_x and audio_x must be 2D or 3D tensors.")
        return text_x, audio_x

    def _topk_edges(self, features, k):
        if k <= 0:
            return None
        num_nodes = features.size(0)
        if num_nodes <= 1:
            return None
        k = min(k, num_nodes - 1)
        feats = F.normalize(features, p=2, dim=1, eps=1e-12)
        sim = feats @ feats.t()
        sim.fill_diagonal_(-float("inf"))
        topk_idx = torch.topk(sim, k=k, dim=1).indices
        row = torch.arange(num_nodes, device=features.device).unsqueeze(1).expand_as(topk_idx)
        return torch.stack([row.reshape(-1), topk_idx.reshape(-1)], dim=0)

    def _build_topk_edge_index(self, text_x, audio_x):
        with torch.no_grad():
            num_nodes = text_x.size(0)
            if num_nodes == 0:
                return torch.empty((2, 0), dtype=torch.long, device=text_x.device)
            self_loops = torch.arange(num_nodes, device=text_x.device)
            edge_parts = [torch.stack([self_loops, self_loops], dim=0)]

            text_edges = self._topk_edges(text_x, self.k_text)
            if text_edges is not None:
                edge_parts.append(text_edges)

            audio_edges = self._topk_edges(audio_x, self.k_audio)
            if audio_edges is not None:
                edge_parts.append(audio_edges)

            edge_index = torch.cat(edge_parts, dim=1)
            edge_index = torch.unique(edge_index, dim=1)
            return edge_index

    def forward(self, text_x, audio_x, edge_index=None, return_all=False):
        text_x, audio_x = self._flatten_modalities(text_x, audio_x)
        if edge_index is None:
            edge_index = self._build_topk_edge_index(text_x, audio_x)

        text_feat = self.text_projection(text_x)
        audio_feat = self.audio_projection(audio_x)

        gnn_input = torch.cat([text_feat, audio_feat], dim=1)
        graph_feat = self.gnn(gnn_input, edge_index)

        CMT_feature = self.cross_fusion(text_feat, audio_feat)

        combined = torch.cat([graph_feat, CMT_feature], dim=1)
        out = self.mlp(combined)
        if return_all:
            return {
                "logits": out,
                "text_proj": text_feat,
                "audio_proj": audio_feat,
                "text_pool": text_feat,
                "audio_pool": audio_feat,
            }
        return out

    def get_fused_embeddings(self, text_x, audio_x):
        text_x, audio_x = self._flatten_modalities(text_x, audio_x)
        text_feat = self.text_projection(text_x)
        audio_feat = self.audio_projection(audio_x)
        CMT_feature = self.cross_fusion(text_feat, audio_feat)
        return CMT_feature
