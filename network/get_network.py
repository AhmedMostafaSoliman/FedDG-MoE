from network import ResNet
import torch.nn as nn
from utils.lora_util import inject_trainable_moe_kronecker_new
import timm
import torch
from typing import Union, List, Tuple


def my_get_intermediate_layers(
        featurizer,
        x: torch.Tensor,
        n: Union[int, List[int], Tuple[int]] = 1,
        return_prefix_tokens: bool = False,
        norm: bool = False,
):
    """
    Extract intermediate layers from a Vision Transformer model
    Adapted From timm's get_intermediate_layers function
    
    Args:
        featurizer: The ViT model to extract features from
        x: Input tensor
        n: Take last n blocks if int, all if None, select matching indices if sequence
        return_prefix_tokens: Return both prefix and spatial intermediate tokens
        norm: Apply norm layer to all intermediates
        
    Returns:
        List of intermediate features, optionally tuples of (spatial_tokens, prefix_tokens)
    """
    # Helper function to determine which indices to take
    def feature_take_indices(num_blocks, indices):
        if indices is None:
            take_indices = list(range(num_blocks))
            max_index = num_blocks - 1
        elif isinstance(indices, int):
            take_indices = list(range(num_blocks))[-indices:]
            max_index = num_blocks - 1
        else:
            take_indices = [i for i in indices if i < num_blocks]
            max_index = max(take_indices) if take_indices else 0
        return take_indices, max_index
    
    intermediates = []
    take_indices, max_index = feature_take_indices(len(featurizer.blocks), n)
    
    # forward pass
    B, _, height, width = x.shape
    x = featurizer.patch_embed(x)
    
    # Apply positional embedding
    if hasattr(featurizer, '_pos_embed'):
        x = featurizer._pos_embed(x)
    elif hasattr(featurizer, 'pos_embed'):
        x = x + featurizer.pos_embed
        
    if hasattr(featurizer, 'patch_drop'):
        x = featurizer.patch_drop(x)
    
    if hasattr(featurizer, 'norm_pre'):
        x = featurizer.norm_pre(x)
    
    # Process through blocks and collect intermediates
    blocks = featurizer.blocks[:max_index + 1]
    for i, blk in enumerate(blocks):
        x = blk(x)
        if i in take_indices:
            # normalize intermediates with final norm layer if enabled
            intermediates.append(featurizer.norm(x) if norm else x)
    
    # process intermediates
    num_prefix_tokens = getattr(featurizer, 'num_prefix_tokens', 1)  # Default to 1 (cls token)
    
    if return_prefix_tokens:
        # split prefix (e.g. class, distill) and spatial feature tokens
        prefix_tokens = [y[:, 0:num_prefix_tokens] for y in intermediates]
        spatial_tokens = [y[:, num_prefix_tokens:] for y in intermediates]
        # Return as list of tuples (spatial_tokens, prefix_tokens)
        return list(zip(spatial_tokens, prefix_tokens))
    
    # If not returning prefix tokens, just return intermediates
    return intermediates


def feats_extractor(x, featurizer, avg_tokens=False, num_layers=4):
    """
    Extract features from a featurizer, ensuring output shape is always [batch_size, feature_dim]
    
    Args:
        x: Input tensor
        featurizer: Feature extractor model
        avg_tokens: Whether to average token features
        num_layers: Number of layers to extract from if mul_layers=True
        
    Returns:
        Tensor of shape [batch_size, feature_dim]
    """
    if num_layers > 1:
        try:
            # Attempt to extract intermediate features using timm's get_intermediate_layers
            intermediate_features = featurizer.get_intermediate_layers(
                x,
                n=num_layers,
                return_prefix_tokens=True,
                norm=True
            )
        except:
            # If failed, use our custom implementation
            intermediate_features = my_get_intermediate_layers(
                featurizer,
                x,
                n=num_layers,
                return_prefix_tokens=True,
                norm=True
            )        
        layers = []
        for spatial_tokens, prefix_tokens in intermediate_features:
            all_tokens = torch.cat([prefix_tokens, spatial_tokens], dim=1)
            if avg_tokens:
                # Average token dimension, preserving [batch_size, embed_dim]
                layers.append(all_tokens.mean(dim=1))
            else:
                # Flatten tokens into a single feature vector per sample
                batch_size = all_tokens.size(0)
                layers.append(all_tokens.reshape(batch_size, -1))
                
        # Average across layers
        final_features = torch.stack(layers).mean(dim=0)
    else:
        # Regular feature extraction
        final_features = featurizer(x)
    
    # Ensure output shape is [batch_size, feature_dim]
    batch_size = final_features.size(0)
    final_features = final_features.reshape(batch_size, -1)
    
    return final_features


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
def GetNetwork(args, num_classes, pretrained=True, **kwargs):
    if args.model == 'resnet18':
        model = ResNet.resnet18(pretrained=pretrained, num_classes=num_classes, **kwargs)
        feature_level = 512
        
    elif args.model == 'resnet18_rsc':
        model = ResNet.resnet18_rsc(pretrained=pretrained, num_classes=num_classes, **kwargs)
        feature_level = 512

    elif args.model == 'resnet50':
        model = ResNet.resnet50(pretrained=pretrained, num_classes=num_classes, **kwargs)
        feature_level = 2048
        
    elif args.model == 'resnet50_rsc':
        model = ResNet.resnet50_rsc(pretrained=pretrained, num_classes=num_classes, **kwargs)
        feature_level = 2048

    elif args.model == 'clip':
        model = timm.create_model('vit_base_patch16_clip_224.laion2b', pretrained=True, num_classes=num_classes)
        feature_level = 768
    
    elif args.model == 'clip_moe':
        featurizer = timm.create_model('vit_base_patch16_clip_224.laion2b', pretrained=True)
        feature_level = featurizer.num_features
        featurizer.head = Identity()
        for name, param in featurizer.named_parameters():
            param.requires_grad = False
        inject_trainable_moe_kronecker_new(featurizer, r=[1, 2, 4, 8], where='every_qkv')
        classifier = nn.Linear(feature_level, num_classes)
        model = nn.Sequential(featurizer, classifier)

    else:
        raise ValueError("The model is not support")

    return model, feature_level
