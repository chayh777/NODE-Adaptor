import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import List, Optional


class TipAdapterF(nn.Module):
    """
    Tip-Adapter-F: Training-free Adaption of CLIP for Few-shot Classification
    
    Reference: Zhang et al., "Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification"
    
    Core idea: Build a feature cache from support set and learn adapter weights.
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 512,
        alpha: float = 1.0,
        beta: float = 2.0,
        cache_temperature: float = 0.07
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.beta = beta
        self.cache_temperature = cache_temperature
        
        self.cache_keys = None
        self.adapter_weights = None
        self.class_names = None
    
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
    
    def build_cache(
        self,
        clip_model,
        support_images: torch.Tensor,
        class_names: List[str],
        prompts: List[str],
        train: bool = False
    ):
        """
        Build feature cache from support set.
        
        Args:
            clip_model: CLIP model
            support_images: (N, K, C, H, W) - support images
            class_names: list of class names
            prompts: list of prompt templates
            train: if True, also build learnable adapter weights
        """
        N, K = support_images.shape[:2]
        
        support_images_flat = support_images.view(-1, *support_images.shape[2:])
        visual_features = self.encode_images(clip_model, support_images_flat)
        
        self.cache_keys = visual_features.view(N, K, self.embed_dim)
        
        if train:
            self.adapter_weights = nn.Parameter(
                torch.randn(N, K, self.embed_dim) * 0.01
            )
        else:
            self.adapter_weights = None
        
        text_features_list = []
        for class_name in class_names:
            class_prompts = [p.format(class_name) for p in prompts]
            class_text_features = self.encode_texts(clip_model, class_prompts)
            text_features_list.append(class_text_features.mean(dim=0))
        
        self.text_features = torch.stack(text_features_list, dim=0)
        self.text_features = F.normalize(self.text_features, p=2, dim=-1)
        
        self.class_names = class_names
    
    def forward(
        self,
        clip_model,
        query_images: torch.Tensor,
        support_images: Optional[torch.Tensor] = None,
        class_names: Optional[List[str]] = None,
        prompts: Optional[List[str]] = None,
        train: bool = False
    ):
        """
        Forward pass for few-shot learning.
        
        Args:
            clip_model: CLIP model
            query_images: (Q, C, H, W) - query images
            support_images: (N, K, C, H, W) - support images (optional if cache built)
            class_names: list of class names
            prompts: list of prompt templates
            train: if True, enable training mode
            
        Returns:
            logits: (Q, N) - classification logits
        """
        if self.cache_keys is None and support_images is not None:
            self.build_cache(clip_model, support_images, class_names, prompts, train)
        
        query_features = self.encode_images(clip_model, query_images)
        
        clip_logits = query_features @ self.text_features.T
        clip_logits = clip_logits / self.cache_temperature
        
        if self.cache_keys is not None:
            N, K, D = self.cache_keys.shape
            cache_keys_flat = self.cache_keys.view(-1, D)
            
            cache_logits = query_features @ cache_keys_flat.T
            cache_logits = cache_logits.view(-1, N, K)
            
            if self.adapter_weights is not None and train:
                adapter_logits = query_features @ self.adapter_weights.view(-1, D).T
                adapter_logits = adapter_logits.view(-1, N, K)
                cache_logits = cache_logits + adapter_logits * self.alpha
            
            cache_logits = cache_logits.max(dim=2)[0]
            
            cache_logits = cache_logits * self.beta
            
            logits = clip_logits + cache_logits
        else:
            logits = clip_logits
        
        return logits
    
    def reset_cache(self):
        """Reset the cache."""
        self.cache_keys = None
        self.adapter_weights = None
        self.text_features = None
        self.class_names = None


class TipAdapterFSimple(nn.Module):
    """
    Simplified Tip-Adapter-F for quick testing without training.
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 512,
        alpha: float = 1.0,
        beta: float = 2.0,
        temperature: float = 0.07
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        
        self.cache_keys = None
        self.text_features = None
    
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
    
    def build_cache(self, clip_model, support_images, class_names, prompts):
        N, K = support_images.shape[:2]
        support_images_flat = support_images.view(-1, *support_images.shape[2:])
        visual_features = self.encode_images(clip_model, support_images_flat)
        self.cache_keys = visual_features.view(N, K, self.embed_dim)
        
        text_features_list = []
        for class_name in class_names:
            class_prompts = [p.format(class_name) for p in prompts]
            class_text_features = self.encode_texts(clip_model, class_prompts)
            text_features_list.append(class_text_features.mean(dim=0))
        
        self.text_features = torch.stack(text_features_list, dim=0)
        self.text_features = F.normalize(self.text_features, p=2, dim=-1)
    
    def forward(self, clip_model, query_images, support_images, class_names, prompts):
        if self.cache_keys is None:
            self.build_cache(clip_model, support_images, class_names, prompts)
        
        query_features = self.encode_images(clip_model, query_images)
        
        clip_logits = query_features @ self.text_features.T / self.temperature
        
        N, K, D = self.cache_keys.shape
        cache_keys_flat = self.cache_keys.view(-1, D)
        cache_logits = query_features @ cache_keys_flat.T
        cache_logits = cache_logits.view(-1, N, K).max(dim=2)[0] * self.beta
        
        logits = clip_logits + cache_logits * self.alpha
        
        return logits
