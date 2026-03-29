import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint

from .prototype_builder import CrossModalPrototype
from .ode_function import ODEFunc


class NODEAdapter(nn.Module):
    """
    NODE-Adapter: Neural ODE for Better Vision-Language Reasoning
    
    Two-stage framework:
    1. Cross-modal Prototype Construction
       - Textual Prototype via CLIP text encoder
       - Visual Prototype via CLIP visual encoder  
       - Adaptive fusion via learnable vector u
    
    2. Neural ODE-based Prototype Optimization
       - dp(t)/dt = f_θ(p(t), t, S)
       - Solve IVP using adjoint method
       - T=30 integration steps
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        ode_steps: int = 30,
        temperature: float = 0.07,
        use_adjoint: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ode_steps = ode_steps
        self.temperature = temperature
        self.use_adjoint = use_adjoint
        
        self.prototype_builder = CrossModalPrototype(num_classes, embed_dim)
        self.ode_func = ODEFunc(embed_dim, num_heads)
        
        self._init_parameters()
    
    def _init_parameters(self):
        nn.init.normal_(self.prototype_builder.u.data, mean=0, std=0.01)
    
    def encode_images(self, clip_model, images: torch.Tensor) -> torch.Tensor:
        """Encode images using CLIP visual encoder."""
        with torch.no_grad():
            image_features = clip_model.encode_image(images)
            image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features
    
    def encode_texts(self, clip_model, texts: list) -> torch.Tensor:
        """Encode texts using CLIP text encoder."""
        import clip
        with torch.no_grad():
            text_tokens = clip.tokenize(texts).to(next(clip_model.parameters()).device)
            text_features = clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features
    
    def build_prototypes(
        self,
        clip_model,
        support_images: torch.Tensor,
        class_names: list,
        prompts: list
    ) -> torch.Tensor:
        """
        Build initial prototypes via cross-modal fusion.
        
        Args:
            clip_model: CLIP model
            support_images: (N, K, C, H, W) - support images
            class_names: list of class names
            prompts: list of prompt templates
            
        Returns:
            P_0: (N, D) - initial prototypes
        """
        N, K = support_images.shape[:2]
        D = self.embed_dim
        
        support_images_flat = support_images.view(-1, *support_images.shape[2:])
        visual_features = self.encode_images(clip_model, support_images_flat)
        visual_features = visual_features.view(N, K, D)
        
        text_features_list = []
        for class_name in class_names:
            class_prompts = [p.format(class_name) for p in prompts]
            class_text_features = self.encode_texts(clip_model, class_prompts)
            text_features_list.append(class_text_features)
        text_features = torch.stack(text_features_list, dim=0)
        
        P_0 = self.prototype_builder(
            text_features=text_features,
            visual_features=visual_features,
            num_shots=K
        )
        
        return P_0, visual_features
    
    def optimize_prototypes(
        self,
        p0: torch.Tensor,
        support_features: torch.Tensor = None,
        return_all: bool = False
    ) -> torch.Tensor:
        """
        Optimize prototypes via Neural ODE.
        
        Args:
            p0: (N, D) - initial prototypes
            support_features: optional (N, K, D) - support features for context
            return_all: if True, return all time steps (T, N, D); if False, return final state (N, D)
            
        Returns:
            p_optimized: (N, D) or (T, N, D) - optimized prototypes
        """
        t = torch.linspace(0, 1, self.ode_steps)
        
        from functools import partial
        if support_features is not None:
            ode_func = partial(self.ode_func, support_set=support_features)
        else:
            ode_func = self.ode_func
        
        if self.use_adjoint:
            p_optimized = odeint_adjoint(
                ode_func,
                p0,
                t,
                method='rk4',
                options={'step_size': 1.0 / self.ode_steps},
                adjoint_params=list(self.ode_func.parameters()) + 
                             list(self.prototype_builder.parameters())
            )
        else:
            from torchdiffeq import odeint
            p_optimized = odeint(
                ode_func,
                p0,
                t,
                method='rk4',
                options={'step_size': 1.0 / self.ode_steps}
            )
        
        if return_all:
            return p_optimized
        else:
            return p_optimized[-1]
    
    def classify(
        self,
        query_features: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Classify query images via nearest prototype.
        
        Args:
            query_features: (Q, D) - query image features
            prototypes: (N, D) - class prototypes
            
        Returns:
            logits: (Q, N) - classification logits
        """
        query_features = F.normalize(query_features, p=2, dim=-1)
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        
        logits = (query_features @ prototypes.T) / self.temperature
        
        return logits
    
    def forward(
        self,
        clip_model,
        support_images: torch.Tensor,
        query_images: torch.Tensor,
        class_names: list,
        prompts: list,
        return_trajectory: bool = False
    ):
        """
        Full forward pass.
        
        Args:
            clip_model: CLIP model
            support_images: (N, K, C, H, W) - support images
            query_images: (Q, C, H, W) - query images
            class_names: list of class names
            prompts: list of prompt templates
            return_trajectory: whether to return ODE trajectory
            
        Returns:
            logits: (Q, N) - classification logits
            (optional) trajectory: (T, N, D) - prototype trajectory
        """
        p0, support_features = self.build_prototypes(
            clip_model=clip_model,
            support_images=support_images,
            class_names=class_names,
            prompts=prompts
        )
        
        p_optimized = self.optimize_prototypes(p0, support_features)
        
        query_features = self.encode_images(clip_model, query_images)
        logits = self.classify(query_features, p_optimized)
        
        if return_trajectory:
            trajectory = self.optimize_prototypes(p0, support_features, return_all=True)
            return logits, trajectory
        else:
            return logits
