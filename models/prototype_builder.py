import torch
import torch.nn as nn
from typing import List, Optional


class CrossModalPrototype(nn.Module):
    """
    Cross-modal Prototype Construction
    
    Stage 1: Build Textual and Visual Prototypes
    - Textual Prototype: P_t = [t̄_1, t̄_2, ..., t̄_N] ∈ ℝ^{N×D}
    - Visual Prototype:  P_v = [v̄_1, v̄_2, ..., v̄_N] ∈ ℝ^{N×D}
    
    Stage 2: Adaptive Fusion
    - p_j = λ_j · v̄_j + (1 - λ_j) · t̄_j
    - λ_j = 1/(1 + exp(-v̄_j^⊤ u))
    """
    
    def __init__(self, num_classes: int, embed_dim: int = 512):
        super().__init__()
        self.N = num_classes
        self.D = embed_dim
        
        self.u = nn.Parameter(torch.randn(embed_dim))
    
    def build_textual_prototype(
        self, 
        text_features: torch.Tensor, 
        num_prompts: int = 20
    ) -> torch.Tensor:
        """
        Build textual prototype by averaging features from M prompt templates.
        
        Args:
            text_features: (N, M, D) - features from M prompts for N classes
            num_prompts: M - number of prompt templates
            
        Returns:
            P_t: (N, D) - textual prototype for each class
        """
        P_t = text_features.mean(dim=1)
        return P_t
    
    def build_visual_prototype(
        self, 
        visual_features: torch.Tensor, 
        num_shots: int
    ) -> torch.Tensor:
        """
        Build visual prototype by averaging features from K support images.
        
        Args:
            visual_features: (N, K, D) - features from K shots for N classes
            num_shots: K - number of support images per class
            
        Returns:
            P_v: (N, D) - visual prototype for each class
        """
        P_v = visual_features.mean(dim=1)
        return P_v
    
    def adaptive_fusion(self, P_t: torch.Tensor, P_v: torch.Tensor) -> torch.Tensor:
        """
        Adaptive fusion of textual and visual prototypes.
        
        Args:
            P_t: (N, D) - textual prototypes
            P_v: (N, D) - visual prototypes
            
        Returns:
            P_0: (N, D) - initial prototypes p(t_0)
        """
        lambda_j = torch.sigmoid(P_v @ self.u)
        lambda_j = lambda_j.unsqueeze(1)
        
        P_0 = lambda_j * P_v + (1 - lambda_j) * P_t
        return P_0
    
    def forward(
        self, 
        text_features: torch.Tensor, 
        visual_features: torch.Tensor,
        num_shots: int,
        num_prompts: int = 20
    ) -> torch.Tensor:
        """
        Full forward pass for cross-modal prototype construction.
        
        Args:
            text_features: (N, M, D) - textual features
            visual_features: (N, K, D) - visual features
            num_shots: K - number of shots
            num_prompts: M - number of prompts
            
        Returns:
            P_0: (N, D) - initial prototypes
        """
        P_t = self.build_textual_prototype(text_features, num_prompts)
        P_v = self.build_visual_prototype(visual_features, num_shots)
        P_0 = self.adaptive_fusion(P_t, P_v)
        
        return P_0
