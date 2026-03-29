import torch
import torch.nn as nn
import torch.nn.functional as F


class ODEFunc(nn.Module):
    """
    Neural ODE Gradient Estimator f_θ(p(t), t, S)
    
    Dual-stage design:
    - Stage 1: Time embedding via MLP
    - Stage 2: Multi-head self-attention (8 Heads) for capturing inter-sample relationships
    
    State evolution: dp(t)/dt = f_θ(p(t), t, S)
    """
    
    def __init__(
        self, 
        embed_dim: int = 512, 
        num_heads: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        self.time_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, p: torch.Tensor, t: torch.Tensor, support_set: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the gradient dp/dt.
        
        Args:
            p: (N, D) - current prototype state
            t: scalar or (1,) - current time point
            support_set: optional (N, K, D) - support set features for context
            
        Returns:
            dpdt: (N, D) - gradient at time t
        """
        if p.numel() == 0 or p.dim() == 0:
            return p
        
        if p.dim() != 2 or p.shape[-1] != self.embed_dim:
            return torch.zeros_like(p)
        
        N, D = p.shape
        
        if isinstance(t, float):
            t = torch.tensor([t], device=p.device, dtype=p.dtype)
        elif t.dim() == 0:
            t = t.unsqueeze(0)
        
        if support_set is not None and support_set.numel() > 0:
            support_pooled = support_set.mean(dim=1)
            support_proj = self.input_proj(support_pooled)
            p_with_context = p + support_proj
        else:
            p_with_context = p
        
        t_emb = self.time_mlp(t.view(1, 1))
        
        time_gate = self.time_gate(t_emb)
        p_with_time = p_with_context * time_gate + t_emb * (1 - time_gate)
        
        p_proj = self.input_proj(p_with_time)
        
        attn_output, _ = self.self_attention(
            p_proj.unsqueeze(0),
            p_proj.unsqueeze(0),
            p_proj.unsqueeze(0)
        )
        attn_output = attn_output.squeeze(0)
        
        attn_output = self.layer_norm1(p_with_time + attn_output)
        
        mlp_output = self.output_mlp(attn_output)
        
        dpdt = self.layer_norm2(attn_output + mlp_output)
        
        dpdt = self.output_proj(dpdt)
        
        return dpdt
