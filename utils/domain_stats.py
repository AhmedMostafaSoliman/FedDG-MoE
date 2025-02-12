import torch
import torch.nn as nn

class DomainStatisticsTracker:
    def __init__(self, feature_dim, num_domains):
        """Initialize domain statistics tracker"""
        self.feature_dim = feature_dim
        self.num_domains = num_domains
        self.register_buffers()
        
    def register_buffers(self):
        """Initialize running statistics buffers on CUDA"""
        self.means = torch.zeros(self.num_domains, self.feature_dim).cuda()
        self.M2 = torch.zeros(self.num_domains, self.feature_dim).cuda()  # Initialize M2
        self.counts = torch.zeros(self.num_domains).cuda()
        
    def get_variance(self, domain_id):
        """Compute variance from M2 statistics"""
        count = self.counts[domain_id]
        if count < 2:
            return torch.ones_like(self.M2[domain_id])
        return self.M2[domain_id] / (count - 1)
    
    def get_domain_weights(self, features):
        """Compute domain similarity weights using Log Likelihood with softmax
        
        Args:
            features: Input feature tensor of shape (batch_size, feature_dim)
            
        Returns:
            domain_weights: Normalized weights for each domain (batch_size, num_domains)
        """
        if not features.is_cuda:
            features = features.cuda()
        
        batch_size = features.size(0)
        log_probs = torch.zeros(batch_size, self.num_domains).cuda()
        
        # Compute log likelihood for each domain
        for domain_id in range(self.num_domains):               
            # Get domain statistics
            domain_mean = self.means[domain_id]
            domain_var = self.get_variance(domain_id)
            
            # Avoid numerical instability
            eps = 1e-6
            domain_var = torch.clamp(domain_var, min=eps)
            
            # Compute log likelihood using Gaussian probability density
            diff = features - domain_mean
            log_probs[:, domain_id] = -0.5 * torch.sum(
                (diff * diff) / domain_var + torch.log(domain_var), 
                dim=1
            )
        
        # Apply softmax to get normalized weights
        domain_weights = torch.softmax(log_probs, dim=1)
        
        return domain_weights
    
    def update(self, features, domain_id):
        """Update running statistics for a domain using Welford's algorithm
        
        Args:
            features: Batch of feature vectors (B, feature_dim) on CUDA
            domain_id: Domain identifier
        """
        with torch.no_grad():
            if not features.is_cuda:
                features = features.cuda()
            
            batch_size = features.size(0)
            old_count = self.counts[domain_id]
            new_count = old_count + batch_size
            
            # Compute batch statistics
            batch_mean = torch.mean(features, dim=0)
            batch_M2 = torch.var(features, dim=0, unbiased=False) * batch_size
            
            # Update mean
            delta = batch_mean - self.means[domain_id]
            self.means[domain_id] += delta * (batch_size / new_count)
            
            # Update M2 (sum of squared deviations)
            self.M2[domain_id] += batch_M2 + delta ** 2 * (old_count * batch_size / new_count)
            
            # Update count
            self.counts[domain_id] = new_count