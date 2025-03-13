from network import ResNet
import torch.nn as nn
from utils.lora_util import inject_trainable_moe_kronecker_new
import timm
import torch


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
        intermediate_features = featurizer.get_intermediate_layers(
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
