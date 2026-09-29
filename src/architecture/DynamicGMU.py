import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGMU(nn.Module):
    def __init__(self, text_dim, audio_dim, fusion_dim, dropout_p=0.2):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.gate_proj = nn.Linear(text_dim + audio_dim, fusion_dim)

        self.q_proj = nn.Linear(fusion_dim, fusion_dim)
        self.k_proj = nn.Linear(fusion_dim, fusion_dim)
        self.v_proj = nn.Linear(fusion_dim, fusion_dim)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, text_feat, audio_feat):
        h_t = self.tanh(self.text_proj(text_feat))
        h_a = self.tanh(self.audio_proj(audio_feat))

        Q = self.q_proj(torch.stack([h_t, h_a], dim=1)) 
        K = self.k_proj(torch.stack([h_t, h_a], dim=1))
        V = self.v_proj(torch.stack([h_t, h_a], dim=1))

        A = torch.softmax(Q @ K.transpose(-2, -1) / (h_t.size(-1) ** 0.5), dim=-1)  

        H_graph = A @ V 
        h_t_graph, h_a_graph = H_graph[:,0,:], H_graph[:,1,:]

        gate_input = torch.cat([text_feat, audio_feat], dim=-1)
        z = self.sigmoid(self.gate_proj(gate_input))

        fused = (1 - z) * h_t_graph + z * h_a_graph
        return fused
