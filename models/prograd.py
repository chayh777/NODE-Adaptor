import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import List, Optional


class ProGrad(nn.Module):
    """
    ProGrad: Prompt-aligned Gradient for Prompt Tuning
    
    Reference: Zhu et al., "Prompt-aligned Gradient for Prompt Tuning"
    
    Core idea: Align learnable prompts with fixed prompts using gradient similarity.
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 512,
        num_prompts: int = 16,
        grad_lambda: float = 0.3,
        temperature: float = 0.07
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_prompts = num_prompts
        self.grad_lambda = grad_lambda
        self.temperature = temperature
        
        self.learnable_prompts = nn.Parameter(torch.randn(num_prompts, embed_dim))
        
        self.scale = nn.Parameter(torch.ones(1) * 2.0)
        
        self._init_parameters()
    
    def _init_parameters(self):
        nn.init.normal_(self.learnable_prompts.data, mean=0, std=0.02)
    
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
    
    def get_fixed_prompts(self, clip_model, class_names: List[str], prompts: List[str]) -> torch.Tensor:
        """
        Get fixed prompt embeddings as reference.
        """
        text_features_list = []
        for class_name in class_names:
            class_prompts = [p.format(class_name) for p in prompts]
            class_text_features = self.encode_texts(clip_model, class_prompts)
            text_features_list.append(class_text_features.mean(dim=0))
        
        text_features = torch.stack(text_features_list, dim=0)
        return F.normalize(text_features, p=2, dim=-1)
    
    def compute_grad_alignment_loss(self, learnable_logits, fixed_logits, labels):
        """
        Compute gradient alignment loss between learnable and fixed prompts.
        
        The idea is to make the gradient of learnable prompts aligned with fixed prompts.
        """
        loss_learnable = F.cross_entropy(learnable_logits, labels)
        loss_fixed = F.cross_entropy(fixed_logits, labels)
        
        return loss_learnable + self.grad_lambda * loss_fixed
    
    def forward(
        self,
        clip_model,
        support_images: torch.Tensor,
        query_images: torch.Tensor,
        class_names: List[str],
        prompts: List[str],
        return_features: bool = False
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
        
        fixed_text_features = self.get_fixed_prompts(clip_model, class_names, prompts)
        
        text_features_list = []
        for class_idx, class_name in enumerate(class_names):
            class_prompts = [p.format(class_name) for p in prompts]
            class_text_features = self.encode_texts(clip_model, class_prompts)
            text_features_list.append(class_text_features.mean(dim=0))
        
        text_features = torch.stack(text_features_list, dim=0)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        learnable_text_features = text_features * self.scale
        
        query_features = self.encode_images(clip_model, query_images)
        
        fixed_logits = query_features @ fixed_text_features.T / self.temperature
        learnable_logits = query_features @ learnable_text_features.T / self.temperature
        
        logits = learnable_logits + fixed_logits
        
        if return_features:
            return logits, query_features, learnable_text_features
        return logits


class ProGradSimple(nn.Module):
    """
    Simplified ProGrad for quick testing - combines learnable and fixed prompts.
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 512,
        temperature: float = 0.07
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.temperature = temperature
        
        self.scale = nn.Parameter(torch.ones(1) * 1.5)
    
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
        """Simple ProGrad: combine fixed prompts with learnable scaling."""
        text_features_list = []
        for class_name in class_names:
            class_prompts = [p.format(class_name) for p in prompts]
            class_text_features = self.encode_texts(clip_model, class_prompts)
            text_features_list.append(class_text_features.mean(dim=0))
        
        text_features = torch.stack(text_features_list, dim=0)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        text_features_scaled = text_features * self.scale
        
        query_features = self.encode_images(clip_model, query_images)
        
        logits = query_features @ text_features_scaled.T / self.temperature
        
        return logits
