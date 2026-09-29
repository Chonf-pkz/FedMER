import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalEncoders(nn.Module):
    def __init__(self, text_input_dim, audio_input_dim, fusion_dim, dropout, num_heads):
        super().__init__()
        self.text_encoder = nn.Sequential(
            nn.Linear(text_input_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout)
        )
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_input_dim, fusion_dim),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout)
        )
        self.self_attention_text = nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attention_audio = nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_text = nn.LayerNorm(fusion_dim)
        self.norm_audio = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feat, audio_feat=None):
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)

        text_encoded = self.text_encoder(text_feat)
        text_attn, _ = self.self_attention_text(text_encoded, text_encoded, text_encoded)
        text_out = self.norm_text(text_encoded + self.dropout(text_attn))

        if audio_feat is None:
            # If text-only mode
            return text_out, None

        if audio_feat.dim() == 2:
            audio_feat = audio_feat.unsqueeze(1)
        audio_encoded = self.audio_encoder(audio_feat)
        audio_attn, _ = self.self_attention_audio(audio_encoded, audio_encoded, audio_encoded)
        audio_out = self.norm_audio(audio_encoded + self.dropout(audio_attn))

        return text_out, audio_out


class FlexibleMMSER(nn.Module):
    def __init__(
        self,
        text_input_dim,
        audio_input_dim,
        num_classes=7,
        fusion_method='self_attention',
        alpha=0.5,
        dropout_rate=0.3,
        use_layernorm=True,
        text_only=False,
        hidden_dim=256,
        proj_dim=64,
        num_heads=4
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.alpha = alpha
        self.text_only = text_only
        norm_layer = nn.LayerNorm if use_layernorm else nn.BatchNorm1d

        # use provided input dims
        self.projection = ModalEncoders(text_input_dim, audio_input_dim, hidden_dim, dropout_rate, num_heads=num_heads)
        self.attn_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.multihead_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn_dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            norm_layer(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(proj_dim // 2, num_classes)
        )

    # ----------------------------- Fuzzy Logic -----------------------------
    def fuzzy_membership(self, x, method='sigmoid'):
        x = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-5)
        if method == 'sigmoid':
            return torch.sigmoid(torch.clamp(x, -4, 4))
        elif method == 'tanh':
            return torch.tanh(torch.clamp(x, -2, 2))
        elif method == 'linear':
            return torch.clamp(F.softplus(x), 0, 1)
        elif method == 'gaussian':
            mean, std = 0.5, 0.15
            x = torch.clamp(x, 0, 1)
            return torch.exp(-((x - mean) ** 2) / (2 * std ** 2))
        elif method == 'piecewise':
            x = torch.clamp(x, 0, 1)
            return torch.where(x < 0.3, 0.2 * x, torch.where(x < 0.7, 0.5 + 0.5 * x, 0.9))
        else:
            raise ValueError(f"Unknown fuzzy membership method: {method}")

    def select_fuzzy_type(self, input_data):
        mean_value = input_data.mean().item()
        std_dev = input_data.std().item()
        skewness = torch.mean((input_data - mean_value) ** 3) / (std_dev ** 3 + 1e-5)
        min_val, max_val = input_data.min().item(), input_data.max().item()
        if -0.5 < skewness < 0.5 and 0 <= min_val < max_val <= 1:
            return 'linear' if std_dev < 0.1 else 'gaussian'
        elif skewness < -0.5 or skewness > 0.5:
            return 'tanh'
        elif mean_value > 0.7 or max_val > 1.5:
            return 'sigmoid'
        else:
            return 'piecewise'

    def fuzzy_fusion(self, text_fuzzy, audio_fuzzy=None):
        if audio_fuzzy is None:
            # text-only fallback
            return text_fuzzy.mean(dim=1)

        concat = torch.cat([text_fuzzy, audio_fuzzy], dim=2)
        projected = self.attn_proj(concat)

        if self.fusion_method == 'self_attention':
            attn_out, _ = self.multihead_attention(projected, projected, projected)
            attn_out = self.attn_dropout(attn_out)
            out = self.attn_norm(attn_out + projected)
            return out.mean(dim=1)

        elif self.fusion_method == 'cross_attention':
            attn_ta, _ = self.multihead_attention(text_fuzzy, audio_fuzzy, audio_fuzzy)
            attn_at, _ = self.multihead_attention(audio_fuzzy, text_fuzzy, text_fuzzy)
            fused = self.alpha * attn_ta + (1 - self.alpha) * attn_at
            fused = self.attn_dropout(fused)
            fused = self.attn_norm(fused + projected)
            return fused.mean(dim=1)

        elif self.fusion_method == 'concat':
            return projected.mean(dim=1)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    # ----------------------------- Forward -----------------------------
    def forward(self, text_embed, audio_embed=None, return_embedding=False, return_all=False):
        # core projection outputs (sequence outputs)
        text_seq, audio_seq = self.projection(text_embed, None if self.text_only else audio_embed)

        # create pooled projections (mean over sequence length)
    
        text_fuzzy_type = self.select_fuzzy_type(text_seq)
        text_fuzzy = self.fuzzy_membership(text_seq, text_fuzzy_type)

        audio_fuzzy_type = self.select_fuzzy_type(audio_seq)
        audio_fuzzy = self.fuzzy_membership(audio_seq, audio_fuzzy_type)
        fused = self.fuzzy_fusion(text_fuzzy, audio_fuzzy)

        logits = self.fc(fused)

        # support two return styles to integrate with existing utils
        if return_embedding:
            return logits, fused

        if return_all:
            return self.as_output_dict(logits, text_seq, audio_seq)

        return logits

    def as_output_dict(self, logits, text_seq, audio_seq):
        """Helper to construct the outputs dict expected by combined losses and utils.train_and_evaluate.

        logits: tensor (B, C)
        text_seq: tensor (B, L, H)
        audio_seq: tensor or None (B, L, H)
        """
        text_proj = text_seq.mean(dim=1)
        audio_proj = audio_seq.mean(dim=1) if audio_seq is not None else torch.zeros_like(text_proj)
        text_pool = text_proj
        audio_pool = audio_proj
        return {
            "logits": logits,
            "text_proj": text_proj,
            "audio_proj": audio_proj,
            "text_pool": text_pool,
            "audio_pool": audio_pool
        }
