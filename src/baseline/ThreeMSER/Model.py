import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeMSER(nn.Module):
    def __init__(
        self,
        text_input_dim,
        audio_input_dim,
        num_classes=4,
        num_attention_head=8,
        dropout=0.5,
        fusion_dim=128,
    ):
        super().__init__()
        self.text_attention = nn.MultiheadAttention(
            embed_dim=text_input_dim,
            num_heads=num_attention_head,
            dropout=dropout,
            batch_first=True,
        )
        self.text_linear = nn.Linear(text_input_dim, fusion_dim)
        self.text_layer_norm = nn.LayerNorm(fusion_dim)

        self.audio_linear = nn.Linear(audio_input_dim, fusion_dim)
        self.audio_layer_norm = nn.LayerNorm(fusion_dim)

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_attention_head,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_linear = nn.Linear(fusion_dim, fusion_dim)
        self.fusion_layer_norm = nn.LayerNorm(fusion_dim)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(fusion_dim, fusion_dim // 2)
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

    def _as_sequence(self, x, name):
        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() == 3:
            return x
        raise ValueError(f"{name} must be 2D or 3D, got shape {tuple(x.shape)}")

    def _as_feature(self, x, name):
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            return x.mean(dim=1)
        raise ValueError(f"{name} must be 2D or 3D, got shape {tuple(x.shape)}")

    def forward(self, text_x, audio_x, return_all=False):
        text_seq = self._as_sequence(text_x, "text_x")
        text_attn, _ = self.text_attention(text_seq, text_seq, text_seq)
        text_proj = self.text_layer_norm(self.text_linear(text_attn))

        audio_feat = self._as_feature(audio_x, "audio_x")
        audio_proj = self.audio_layer_norm(self.audio_linear(audio_feat))
        audio_seq = audio_proj.unsqueeze(1)

        fusion_seq = torch.cat([text_proj, audio_seq], dim=1)
        fusion_attn, _ = self.fusion_attention(fusion_seq, fusion_seq, fusion_seq)
        fusion_proj = self.fusion_layer_norm(self.fusion_linear(fusion_attn))

        cls_token = fusion_proj[:, 0, :]
        x = self.dropout(cls_token)
        x = self.linear(x)
        x = F.leaky_relu(x)
        logits = self.classifier(x)

        if return_all:
            text_pool = text_proj.mean(dim=1)
            return {
                "logits": logits,
                "text_proj": text_pool,
                "audio_proj": audio_proj,
                "text_pool": text_pool,
                "audio_pool": audio_proj,
            }

        return logits
