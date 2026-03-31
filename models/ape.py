import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import List, Optional


class APE(nn.Module):
    """
    APE: Adaptive Prior Refinement for Few-shot CLIP
    
    Reference: Zhu et al., "Not All Features Matter: Enhancing Few-shot CLIP 
               with Adaptive Prior Refinement"
    
    Core idea: Multiple expert prompts with adaptive fusion weights.
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 512,
        num_experts: int = 8,
        num_prompts: int = 16,
        temperature: float = 0.07
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.num_prompts = num_prompts
        self.temperature = temperature
        
        self.expert_prompts = nn.Parameter(
            torch.randn(num_experts, num_prompts, embed_dim) * 0.02
        )
        
        self.fusion_weights = nn.Parameter(torch.ones(num_experts) / num_experts)
        
        self.prior_refinement = PriorRefinement(embed_dim)
        
        self._init_parameters()
    
    def _init_parameters(self):
        nn.init.normal_(self.expert_prompts.data, mean=0, std=0.02)
        nn.init.normal_(self.fusion_weights.data, mean=0, std=0.01)
    
    def encode_images(self, clip_model, images: torch.Tensor) -> torch.Tensor:
        """Encode images using CLIP visual encoder."""
        with torch.no_grad():
            image_features = clip_model.encode_image(images)
            image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features
    
    def encode_texts(self, clip_model, texts: List[str]) -> torch.Tensor:
        """Encode texts using CLIP text encoder."""
        with torch.no_grad():
            text_tokens = clip.tokenize(texts).to(next(clip_model.parameters()).device)
            text_features = clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features
    
    def get_expert_text_features(
        self,
        clip_model,
        class_names: List[str],
        prompts: List[str]
    ) -> torch.Tensor:
        """
        Get text features from multiple expert prompts.
        
        Returns:
            expert_features: (num_experts, N, D)
        """
        N = len(class_names)
        
        text_features_list = []
        for class_name in class_names:
            class_prompts = [p.format(class_name) for p in prompts]
            class_text_features = self.encode_texts(clip_model, class_prompts)
            text_features_list.append(class_text_features.mean(dim=0))
        
        fixed_features = torch.stack(text_features_list, dim=0)
        fixed_features = F.normalize(fixed_features, p=2, dim=-1)
        
        expert_features_list = []
        for expert_idx in range(self.num_experts):
            expert_feat = fixed_features + self.prior_refinement(
                self.expert_prompts[expert_idx].mean(dim=0)
            )
            expert_features_list.append(expert_feat)
        
        expert_features = torch.stack(expert_features_list, dim=0)
        
        return expert_features
    
    def forward(
        self,
        clip_model,
        support_images: torch.Tensor,
        query_images: torch.Tensor,
        class_names: List[str],
        prompts: List[str],
        return_weights: bool = False
    ):
        """
        Forward pass for few-shot learning.
        
        Args:
            clip_model: CLIP model
            support_images: (N, K, C, H, W) - support images
            query_images: (Q, C, H, W) - query images
            class_names: list of class names
            prompts: list of prompt templates
            
        Returns:
            logits: (Q, N) - classification logits
        """
        N, K = support_images.shape[:2]
        
        expert_features = self.get_expert_text_features(clip_model, class_names, prompts)
        
        fusion_weights = F.softmax(self.fusion_weights, dim=0)
        
        fused_text_features = (expert_features * fusion_weights.view(-1, 1, 1)).sum(dim=0)
        fused_text_features = F.normalize(fused_text_features, p=2, dim=-1)
        
        query_features = self.encode_images(clip_model, query_images)
        
        logits = query_features @ fused_text_features.T / self.temperature
        
        if return_weights:
            return logits, fusion_weights
        return logits


class PriorRefinement(nn.Module):
    """
    Prior refinement module for adaptive prompt learning.
    """
    
    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class APESimple(nn.Module):
    """
    Simplified APE for quick testing - uses multiple fixed prompts with learnable weights.
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 512,
        num_experts: int = 4,
        temperature: float = 0.07
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.temperature = temperature
        
        self.fusion_weights = nn.Parameter(torch.ones(num_experts) / num_experts)
    
    def encode_images(self, clip_model, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image_features = clip_model.encode_image(images)
            image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features
    
    def encode_texts(self, clip_model, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            text_tokens = clip.tokenize(texts).to(next(clip_model.parameters()).device)
            text_features = clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features
    
    def forward(
        self,
        clip_model,
        support_images: torch.Tensor,
        query_images: torch.Tensor,
        class_names: List[str],
        prompts: List[str]
    ):
        """Simple APE: multiple prompt templates with learnable fusion."""
        N = len(class_names)
        
        all_expert_features = []
        for _ in range(self.num_experts):
            text_features_list = []
            for class_name in class_names:
                class_prompts = [p.format(class_name) for p in prompts]
                class_text_features = self.encode_texts(clip_model, class_prompts)
                text_features_list.append(class_text_features.mean(dim=0))
            
            text_features = torch.stack(text_features_list, dim=0)
            text_features = F.normalize(text_features, p=2, dim=-1)
            all_expert_features.append(text_features)
        
        expert_features = torch.stack(all_expert_features, dim=0)
        
        fusion_weights = F.softmax(self.fusion_weights, dim=0)
        
        fused_features = (expert_features * fusion_weights.view(-1, 1, 1)).sum(dim=0)
        fused_features = F.normalize(fused_features, p=2, dim=-1)
        
        query_features = self.encode_images(clip_model, query_images)
        
        logits = query_features @ fused_features.T / self.temperature
        
        return logits
