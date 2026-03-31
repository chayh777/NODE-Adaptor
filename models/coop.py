import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import List, Optional


class CoOp(nn.Module):
    """
    CoOp: Contrastive Prompt Learning for Vision-Language Models
    
    Reference: Zhou et al., "CoOp: Prompt Learning for Vision-Language Models"
    
    Core idea: Learnable soft prompts that can be optimized for downstream tasks.
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 512,
        num_prompts: int = 16,
        prompt_depth: int = 1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_prompts = num_prompts
        
        self.prompt_learner = PromptLearner(num_classes, embed_dim, num_prompts, prompt_depth)
        
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
    
    def forward(
        self,
        clip_model,
        support_images: torch.Tensor,
        query_images: torch.Tensor,
        class_names: List[str],
        prompts: List[str],
        return_prompts: bool = False
    ):
        """
        Full forward pass for few-shot learning.
        
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
        Q = query_images.shape[0]
        
        prompt_tokens = self.prompt_learner(class_names)
        
        text_features_list = []
        for class_idx, class_name in enumerate(class_names):
            class_prompts = [p.format(class_name) for p in prompts]
            class_text_features = self.encode_texts(clip_model, class_prompts)
            text_features_list.append(class_text_features.mean(dim=0))
        
        text_features = torch.stack(text_features_list, dim=0)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        query_features = self.encode_images(clip_model, query_images)
        
        logits = (query_features @ text_features.T) / 0.07
        
        if return_prompts:
            return logits, prompt_tokens
        return logits
    
    def get_prompt_texts(self, class_names: List[str], device: torch.device) -> List[str]:
        """Generate prompt texts for class names."""
        return [f"a photo of a {name}." for name in class_names]


class PromptLearner(nn.Module):
    """
    Learnable prompt tokens that can be combined with class names.
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        num_prompts: int,
        prompt_depth: int
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_prompts = num_prompts
        
        if prompt_depth == 1:
            self.prompt_embeddings = nn.Parameter(torch.randn(num_prompts, embed_dim))
        else:
            layers = []
            for i in range(prompt_depth):
                if i == 0:
                    layers.append(nn.Linear(embed_dim, embed_dim))
                else:
                    layers.append(nn.GELU())
                    layers.append(nn.Linear(embed_dim, embed_dim))
            self.prompt_mlp = nn.Sequential(*layers)
            self.prompt_embeddings = nn.Parameter(torch.randn(num_prompts, embed_dim))
        
        self.prompt_depth = prompt_depth
        self._init_parameters()
    
    def _init_parameters(self):
        nn.init.normal_(self.prompt_embeddings.data, mean=0, std=0.02)
    
    def forward(self, class_names: List[str]) -> torch.Tensor:
        """
        Generate prompt embeddings for each class.
        
        Args:
            class_names: list of class names
            
        Returns:
            prompt_embeddings: (N, num_prompts, D)
        """
        if self.prompt_depth == 1:
            prompts = self.prompt_embeddings.unsqueeze(0).repeat(len(class_names), 1, 1)
        else:
            prompts = self.prompt_mlp(self.prompt_embeddings.unsqueeze(0))
            prompts = prompts.repeat(len(class_names), 1, 1)
        
        return prompts


class CoOpSimple(nn.Module):
    """
    Simplified CoOp for quick testing - uses fixed prompt templates with learnable scaling.
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
        
        self.scale = nn.Parameter(torch.ones(1) * 2.0)
    
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
        """Simple CoOp: use fixed prompts with learnable scaling."""
        text_features_list = []
        for class_name in class_names:
            class_prompts = [p.format(class_name) for p in prompts]
            class_text_features = self.encode_texts(clip_model, class_prompts)
            text_features_list.append(class_text_features.mean(dim=0))
        
        text_features = torch.stack(text_features_list, dim=0)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        query_features = self.encode_images(clip_model, query_images)
        
        logits = (query_features @ text_features.T) * self.scale
        
        return logits
