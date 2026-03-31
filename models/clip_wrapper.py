"""
CLIP wrapper that supports both openai/clip and transformers CLIP.
"""
import torch
import torch.nn.functional as F
from typing import Optional, List


class CLIPWrapper:
    """
    Wrapper for CLIP model that provides unified interface.
    Supports both openai/clip and transformers CLIP.
    """
    
    def __init__(self, clip_model, model_type: str = 'openai'):
        self.clip_model = clip_model
        self.model_type = model_type
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to feature vectors."""
        if self.model_type == 'transformers':
            with torch.no_grad():
                outputs = self.clip_model.get_image_features(pixel_values=images)
                features = outputs / outputs.norm(dim=-1, keepdim=True)
            return features
        else:
            with torch.no_grad():
                features = self.clip_model.encode_image(images)
                features = F.normalize(features, p=2, dim=-1)
            return features
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to feature vectors."""
        if self.model_type == 'transformers':
            from transformers import CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-b-32")
            text_inputs = tokenizer(texts, padding=True, return_tensors="pt", truncation=True)
            text_inputs = {k: v.to(self.clip_model.device) for k, v in text_inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model.get_text_features(**text_inputs)
                features = outputs / outputs.norm(dim=-1, keepdim=True)
            return features
        else:
            import clip
            with torch.no_grad():
                text_tokens = clip.tokenize(texts).to(next(self.clip_model.parameters()).device)
                features = self.clip_model.encode_text(text_tokens)
                features = F.normalize(features, p=2, dim=-1)
            return features
    
    @property
    def device(self):
        """Get model device."""
        if self.model_type == 'transformers':
            return self.clip_model.device
        else:
            return next(self.clip_model.parameters()).device


def load_clip(model_name: str = "ViT-B/32", device: str = "cuda", use_transformers: bool = False):
    """
    Load CLIP model.
    
    Args:
        model_name: CLIP model name (e.g., "ViT-B/32", "RN101")
        device: device to load model to
        use_transformers: if True, use transformers CLIP instead of openai/clip
        
    Returns:
        clip_model: CLIP model
        preprocess: preprocessing transform (for openai/clip)
    """
    if use_transformers:
        from transformers import CLIPModel, CLIPProcessor
        
        model_name_map = {
            "ViT-B/32": "openai/clip-vit-b-32",
            "ViT-B/16": "openai/clip-vit-b-16",
            "ViT-L-14": "openai/clip-vit-l-14",
        }
        
        hf_name = model_name_map.get(model_name, model_name)
        print(f"Loading CLIP model from transformers: {hf_name}")
        
        clip_model = CLIPModel.from_pretrained(hf_name).to(device)
        clip_model.eval()
        
        processor = CLIPProcessor.from_pretrained(hf_name)
        
        return CLIPWrapper(clip_model, model_type='transformers'), processor
    
    else:
        import clip
        
        print(f"Loading CLIP model: {model_name}")
        clip_model, preprocess = clip.load(model_name, device=device)
        clip_model.eval()
        
        return clip_model, preprocess


# Helper function for tokenization (openai/clip style)
def tokenize(texts: List[str], clip_model) -> torch.Tensor:
    """Tokenize texts in CLIP format."""
    import clip
    return clip.tokenize(texts)