import torch
from .domain_stats import DomainStatisticsTracker
from .moe_utils import MoEParamManager

def test_with_domain_adaptation(model, inputs, domain_stats: DomainStatisticsTracker, 
                              moe_manager: MoEParamManager, train_domains):
    """Perform test-time inference with domain-adaptive MoE routing"""
    with torch.no_grad():
        # Ensure inputs are on CUDA
        if not inputs.is_cuda:
            inputs = inputs.cuda()
            
        # Extract features from frozen CLIP backbone
        features = model[0](inputs)
        
        # Get domain similarity weights
        domain_weights = domain_stats.get_domain_weights(features)

        # Get weighted MoE parameters
        weighted_params = moe_manager.compute_weighted_params(domain_weights, train_domains)

        # Update model's MoE parameters
        moe_manager.load_weighted_params(model, weighted_params)
        
        # Mix MoE parameters based on feature similarity
        weighted_params = moe_manager.compute_weighted_params(domain_weights, train_domains)
        
        # Forward pass with adapted parameters
        outputs = model(inputs)
        
    return outputs, domain_weights