import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Union


class AvgPoolGate(nn.Module):
    def __init__(self, d_model: int, gate_out_dim: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.gate_out_dim = gate_out_dim or d_model
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, self.gate_out_dim)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def masked_avgpool_time(x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if key_padding_mask is None:
            return x.mean(dim=1)

        valid = (~key_padding_mask).to(dtype=x.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (x * valid.unsqueeze(-1)).sum(dim=1) / denom
        return pooled

    def forward(self, h: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        pooled = self.masked_avgpool_time(h, key_padding_mask)
        pooled = self.norm(pooled)
        g = self.fc(pooled)
        g = self.sigmoid(g)
        return g.unsqueeze(1)


class GatedCrossMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_text: int,
        d_audio: int,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        bidirectional: bool = True,
        return_attn: bool = False,
        gate_channelwise: bool = True,
        use_gate: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.return_attn = return_attn
        self.use_gate = use_gate

        self.text_proj = nn.Linear(d_text, d_model)
        self.audio_proj = nn.Linear(d_audio, d_model)

        self.t2a_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        if bidirectional:
            self.a2t_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
            )
        else:
            self.a2t_attn = None

        gate_dim = d_model if gate_channelwise else 1
        if self.use_gate:
            self.t_gate = AvgPoolGate(d_model=d_model, gate_out_dim=gate_dim)
            self.a_gate = AvgPoolGate(d_model=d_model, gate_out_dim=gate_dim) if bidirectional else None
        else:
            self.t_gate = None
            self.a_gate = None

        self.drop = nn.Dropout(dropout)
        self.t_norm = nn.LayerNorm(d_model)
        self.a_norm = nn.LayerNorm(d_model) if bidirectional else None

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        text_key_padding_mask: Optional[torch.Tensor] = None,
        audio_key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask_t2a: Optional[torch.Tensor] = None,
        attn_mask_a2t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:

        if text.dim() == 2:
            text = text.unsqueeze(1)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        T = self.text_proj(text)
        A = self.audio_proj(audio)

        T_ctx, w_t2a = self.t2a_attn(
            query=T,
            key=A,
            value=A,
            key_padding_mask=audio_key_padding_mask,
            attn_mask=attn_mask_t2a,
            need_weights=self.return_attn,
            average_attn_weights=False,
        )

        if self.use_gate:
            g_t = self.t_gate(T_ctx, key_padding_mask=text_key_padding_mask)
            T_out = self.t_norm(T + g_t * self.drop(T_ctx))
        else:
            T_out = self.t_norm(T + self.drop(T_ctx))

        if not self.bidirectional:
            attn_dict = {"t2a": w_t2a} if self.return_attn else None
            return T_out, A, attn_dict

        A_ctx, w_a2t = self.a2t_attn(
            query=A,
            key=T,
            value=T,
            key_padding_mask=text_key_padding_mask,
            attn_mask=attn_mask_a2t,
            need_weights=self.return_attn,
            average_attn_weights=False,
        )

        if self.use_gate:
            g_a = self.a_gate(A_ctx, key_padding_mask=audio_key_padding_mask)
            A_out = self.a_norm(A + g_a * self.drop(A_ctx))
        else:
            A_out = self.a_norm(A + self.drop(A_ctx))

        attn_dict = None
        if self.return_attn:
            attn_dict = {"t2a": w_t2a, "a2t": w_a2t}

        return T_out, A_out, attn_dict


class CrossModalEncoders(nn.Module):
    def __init__(
        self,
        d_text: int,
        d_audio: int,
        d_model: int = 256,
        dropout: float = 0.1,
        num_heads: int = 8,
        num_layers: int = 1,
        bidirectional: bool = True,
        return_attn: bool = False,
        gate_channelwise: bool = True,
        use_gate: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.return_attn = return_attn
        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            layer_text_dim = d_text if idx == 0 else d_model
            layer_audio_dim = d_audio if idx == 0 else d_model
            self.layers.append(
                GatedCrossMultiHeadAttention(
                    d_text=layer_text_dim,
                    d_audio=layer_audio_dim,
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    return_attn=return_attn,
                    gate_channelwise=gate_channelwise,
                    use_gate=use_gate,
                )
            )

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        text_key_padding_mask: Optional[torch.Tensor] = None,
        audio_key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask_t2a: Optional[torch.Tensor] = None,
        attn_mask_a2t: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, List[Optional[Dict[str, torch.Tensor]]]],
    ]:
        if text.dim() == 2:
            text = text.unsqueeze(1)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        attn_records = [] if self.return_attn else None

        for layer in self.layers:
            text, audio, attn = layer(
                text=text,
                audio=audio,
                text_key_padding_mask=text_key_padding_mask,
                audio_key_padding_mask=audio_key_padding_mask,
                attn_mask_t2a=attn_mask_t2a,
                attn_mask_a2t=attn_mask_a2t,
            )
            if self.return_attn:
                attn_records.append(attn)

        if self.return_attn:
            return text, audio, attn_records
        return text, audio
