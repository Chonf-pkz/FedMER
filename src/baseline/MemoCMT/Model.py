from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import List, Optional


@dataclass
class Config:
    text_encoder_dim: int = 768
    audio_encoder_dim: int = 768
    num_attention_head: int = 4
    dropout: float = 0.2
    fusion_dim: int = 512
    linear_layer_output: Optional[List[int]] = None
    num_classes: int = 4
    fusion_head_output_type: str = 'mean'



class MemoCMT(nn.Module):
    def __init__(self, cfg: Config, device: str = "cpu"):
        super(MemoCMT, self).__init__()
        # Fusion module (expecting input tensors already encoded into embeddings)
        self.text_attention = nn.MultiheadAttention(
            embed_dim=cfg.text_encoder_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.text_linear = nn.Linear(cfg.text_encoder_dim, cfg.fusion_dim)
        self.text_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.audio_attention = nn.MultiheadAttention(
            embed_dim=cfg.audio_encoder_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.audio_linear = nn.Linear(cfg.audio_encoder_dim, cfg.fusion_dim)
        self.audio_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=cfg.fusion_dim,
            num_heads=cfg.num_attention_head,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.fusion_linear = nn.Linear(cfg.fusion_dim, cfg.fusion_dim)
        self.fusion_layer_norm = nn.LayerNorm(cfg.fusion_dim)

        self.dropout = nn.Dropout(cfg.dropout)

        self.linear_layer_output = cfg.linear_layer_output or []

        previous_dim = cfg.fusion_dim
        if len(self.linear_layer_output) > 0:
            for i, linear_layer in enumerate(self.linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)

        self.fusion_head_output_type = cfg.fusion_head_output_type

    def forward(self, input_text: torch.Tensor, input_audio: torch.Tensor, return_all: bool = False, output_attentions: bool = False):
        # Expect inputs to already be embeddings.
        # Text: accept (B, H) or (B, L, H)
        if input_text is None:
            raise ValueError("input_text must be provided as embeddings")
        if input_text.dim() == 2:
            text_embeddings = input_text.unsqueeze(1)
        elif input_text.dim() == 3:
            text_embeddings = input_text
        else:
            raise ValueError("input_text must be 2D (B,H) or 3D (B,L,H)")

        # Audio: accept (B, H) or (B, L, H)
        if input_audio is None:
            raise ValueError("input_audio must be provided as embeddings")
        if input_audio.dim() == 2:
            audio_embeddings = input_audio.unsqueeze(1)
        elif input_audio.dim() == 3:
            audio_embeddings = input_audio
        else:
            raise ValueError("input_audio must be 2D (B,H) or 3D (B,L,H)")

        # Text cross attention: Q=audio, K=V=text
        text_attention, text_attn_output_weights = self.text_attention(
            audio_embeddings, text_embeddings, text_embeddings, average_attn_weights=False
        )
        text_linear = self.text_linear(text_attention)
        text_norm = self.text_layer_norm(text_linear)
        text_norm = self.dropout(text_norm)

        # Audio cross attention: Q=text, K=V=audio
        audio_attention, audio_attn_output_weights = self.audio_attention(
            text_embeddings, audio_embeddings, audio_embeddings, average_attn_weights=False
        )
        audio_linear = self.audio_linear(audio_attention)
        audio_norm = self.audio_layer_norm(audio_linear)
        audio_norm = self.dropout(audio_norm)

        fusion_norm = torch.cat((text_norm, audio_norm), dim=1)
        fusion_norm = self.dropout(fusion_norm)

        if self.fusion_head_output_type == "cls":
            cls_token_final_fusion_norm = fusion_norm[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            cls_token_final_fusion_norm = fusion_norm.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            cls_token_final_fusion_norm = fusion_norm.max(dim=1)[0]
        elif self.fusion_head_output_type == "min":
            cls_token_final_fusion_norm = fusion_norm.min(dim=1)[0]
        else:
            raise ValueError("Invalid fusion head output type")

        x = cls_token_final_fusion_norm
        x = self.dropout(x)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        out = self.classifer(x)

        if return_all:
            text_proj = text_norm.mean(dim=1)
            audio_proj = audio_norm.mean(dim=1)
            return {
                "logits": out,
                "text_proj": text_proj,
                "audio_proj": audio_proj,
                "text_pool": text_proj,
                "audio_pool": audio_proj,
            }

        if output_attentions:
            return [out, cls_token_final_fusion_norm], [text_attn_output_weights, audio_attn_output_weights]

        return out, cls_token_final_fusion_norm, text_norm, audio_norm

    def encode_audio(self, audio: torch.Tensor):
        # Encoder removed: assume input is already embedding
        return audio

    def encode_text(self, input_ids: torch.Tensor):
        # Encoder removed: assume input is already embedding
        return input_ids
