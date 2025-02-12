from network import ResNet
import torch.nn as nn
from utils.lora_util import inject_trainable_moe_kronecker_new
import timm
import torch


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
