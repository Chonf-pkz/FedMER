import torch
import torch.nn as nn
import torch.nn.functional as F
class DualHypergraphModule(nn.Module):
    def __init__(
        self,
        dim,
        K_text=10,
        K_audio=10,
        normalize=True,
        proj_hidden=None,
        threshold=None,
        threshold_text=None,
        threshold_audio=None,
    ):
        super().__init__()
        self.K_text = K_text
        self.K_audio = K_audio
        self.normalize = normalize
        self.threshold_text = threshold_text if threshold_text is not None else threshold
        self.threshold_audio = threshold_audio if threshold_audio is not None else threshold

        if proj_hidden is None:
            self.post_proj = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU()
            )
        else:
            self.post_proj = nn.Sequential(
                nn.Linear(dim, proj_hidden),
                nn.ReLU(),
                nn.Linear(proj_hidden, dim),
                nn.LayerNorm(dim),
                nn.GELU()
            )

        self.edge_stats = []

    def _build_intra_hyper_all(self, encoded):
        return encoded.mean(dim=1, keepdim=True)

    @staticmethod
    def _cosine_similarity(a, b):
        a_n = F.normalize(a, p=2, dim=-1)
        b_n = F.normalize(b, p=2, dim=-1)
        return torch.bmm(a_n, b_n.transpose(1, 2))

    def forward(self, text_encoded, audio_encoded, record_stats=True):
        if text_encoded.dim() == 2:
            text_encoded = text_encoded.unsqueeze(1)
        if audio_encoded.dim() == 2:
            audio_encoded = audio_encoded.unsqueeze(1)

        B, Lt, D = text_encoded.shape
        B, La, D = audio_encoded.shape

        if self.normalize:
            text_norm = F.normalize(text_encoded, p=2, dim=-1)
            audio_norm = F.normalize(audio_encoded, p=2, dim=-1)
        else:
            text_norm, audio_norm = text_encoded, audio_encoded

        hyper_text_intra = self._build_intra_hyper_all(text_encoded)
        hyper_audio_intra = self._build_intra_hyper_all(audio_encoded)

        sim_t2a = torch.bmm(text_norm, audio_norm.transpose(1, 2))  # (B, Lt, La)
        if self.threshold_text is not None:
            mask_t2a = (sim_t2a >= self.threshold_text).float()
            neighbor_sum = torch.bmm(mask_t2a, audio_encoded)
            neighbor_count = mask_t2a.sum(dim=-1, keepdim=True)
            hyper_text_inter = (text_encoded + neighbor_sum) / (neighbor_count + 1.0)
        else:
            Kt = min(self.K_text, La)
            _, topk_idx_a = sim_t2a.topk(Kt, dim=-1)
            b_idx = torch.arange(B, device=text_encoded.device).view(B, 1, 1).expand(-1, Lt, Kt)
            gathered_a = audio_encoded[b_idx, topk_idx_a]
            gathered_tcenter = torch.cat([text_encoded.unsqueeze(2), gathered_a], dim=2)
            hyper_text_inter = gathered_tcenter.mean(dim=2)

        sim_a2t = torch.bmm(audio_norm, text_norm.transpose(1, 2))  # (B, La, Lt)
        if self.threshold_audio is not None:
            mask_a2t = (sim_a2t >= self.threshold_audio).float()
            neighbor_sum = torch.bmm(mask_a2t, text_encoded)
            neighbor_count = mask_a2t.sum(dim=-1, keepdim=True)
            hyper_audio_inter = (audio_encoded + neighbor_sum) / (neighbor_count + 1.0)
        else:
            Ka = min(self.K_audio, Lt)
            _, topk_idx_t = sim_a2t.topk(Ka, dim=-1)
            b_idx = torch.arange(B, device=audio_encoded.device).view(B, 1, 1).expand(-1, La, Ka)
            gathered_t = text_encoded[b_idx, topk_idx_t]
            gathered_acenter = torch.cat([audio_encoded.unsqueeze(2), gathered_t], dim=2)
            hyper_audio_inter = gathered_acenter.mean(dim=2)

        hyper_text = (hyper_text_intra + hyper_text_inter) / 2
        hyper_audio = (hyper_audio_intra + hyper_audio_inter) / 2

        hyper_text = self.post_proj(hyper_text)
        hyper_audio = self.post_proj(hyper_audio)

        if record_stats:
            with torch.no_grad():
                def compute_pct_from_mask(mask, nodes_other):
                    B, Nsrc, Nother = mask.shape
                    total_possible = Nsrc * Nother
                    pct_list = []
                    for b in range(B):
                        num_edges = mask[b].sum().item()
                        pct = (num_edges / total_possible) * 100.0 if total_possible > 0 else 0.0
                        pct_list.append(pct)
                    return sum(pct_list) / len(pct_list) if len(pct_list) > 0 else 0.0

                if self.threshold_text is not None:
                    mask_t = (sim_t2a >= self.threshold_text).float()
                    text_edge_pct = compute_pct_from_mask(mask_t, La)
                else:
                    Kt = min(self.K_text, La)
                    text_edge_pct = (Kt / (La)) * 100.0 if La > 0 else 0.0

                if self.threshold_audio is not None:
                    mask_a = (sim_a2t >= self.threshold_audio).float()
                    audio_edge_pct = compute_pct_from_mask(mask_a, Lt)
                else:
                    Ka = min(self.K_audio, Lt)
                    audio_edge_pct = (Ka / (Lt)) * 100.0 if Lt > 0 else 0.0

                stat = {
                    "threshold_text": self.threshold_text,
                    "threshold_audio": self.threshold_audio,
                    "text_edge_pct": text_edge_pct,
                    "audio_edge_pct": audio_edge_pct,
                    "Lt": Lt,
                    "La": La,
                    "batch_size": B
                }
                self.edge_stats.append(stat)

        return hyper_text, hyper_audio