import torch
import torch.nn as nn
import torch.nn.functional as F

class OfflineCosineTracker:
    def __init__(self, num_domains, feature_dim=None, flatten_tokens=False):
        """
        Initialize offline domain statistics tracker using cosine similarity.
        
        Args:
            num_domains: Number of domains to track
            feature_dim: Feature dimension (optional, will be determined from data if None)
            flatten_tokens: Whether to flatten token-level features or average them
        """
        self.num_domains = num_domains
        self.feature_dim = feature_dim
        self.flatten_tokens = flatten_tokens
        self.means = None  # Will be initialized on first fit
        self.fitted = [False] * self.num_domains  # Flag

    def refit(self, features, domain_id):
        """
        Refit the mean (centroid) from scratch.

        Args:
            features: All features for this domain, shape [batch_size, ...] or [batch_size, num_tokens, embed_dim]
            domain_id: Domain identifier.
        """
        with torch.no_grad():
            if not features.is_cuda:
                features = features.cuda()
            
            # Handle first fit - determine dimensions and initialize storage
            if self.means is None:
                if len(features.shape) == 3 and self.flatten_tokens:
                    # Handle token-level features with flattening
                    batch_size, num_tokens, embed_dim = features.shape
                    self.feature_dim = num_tokens * embed_dim
                    features = features.reshape(batch_size, -1)
                elif len(features.shape) == 3 and not self.flatten_tokens:
                    # Average token embeddings
                    batch_size, num_tokens, embed_dim = features.shape
                    self.feature_dim = embed_dim
                    features = features.mean(dim=1)
                else:
                    # Simple feature vector
                    self.feature_dim = features.shape[1]
                
                self.means = torch.zeros(self.num_domains, self.feature_dim).cuda()
            
            # Process features according to settings
            elif len(features.shape) == 3:
                if self.flatten_tokens:
                    # Flatten token dimension into feature dimension
                    batch_size, num_tokens, embed_dim = features.shape
                    features = features.reshape(batch_size, -1)
                else:
                    # Average token embeddings
                    features = features.mean(dim=1)
            
            # Calculate mean
            self.means[domain_id] = torch.mean(features, dim=0)
            self.fitted[domain_id] = True

    def get_domain_weights(self, features):
        """
        Compute domain similarity weights (same as online version).
        """
        if self.means is None:
            raise RuntimeError("Tracker not fitted. Call refit() first.")
            
        if not features.is_cuda:
            features = features.cuda()
            
        # Process features according to settings
        if len(features.shape) == 3:
            if self.flatten_tokens:
                # Flatten token dimension into feature dimension
                batch_size, num_tokens, embed_dim = features.shape
                features = features.reshape(batch_size, -1)
            else:
                # Average token embeddings
                features = features.mean(dim=1)
                
        batch_size = features.size(0)
        similarities = torch.zeros(batch_size, self.num_domains).cuda()

        for domain_id in range(self.num_domains):
            if self.fitted[domain_id]:
              domain_mean = self.means[domain_id]
              similarities[:, domain_id] = F.cosine_similarity(features, domain_mean.unsqueeze(0), dim=1)
            else:
              similarities[:, domain_id] = torch.zeros(batch_size).cuda()

        domain_weights = torch.softmax(similarities, dim=1)
        return domain_weights

    def save_cosine(self, filepath):
        """Save the tracker's parameters."""
        torch.save({
            'means': self.means, 
            'fitted': self.fitted,
            'feature_dim': self.feature_dim,
            'flatten_tokens': self.flatten_tokens
        }, filepath)

    def load_cosine(self, filepath):
        """Load the tracker's parameters."""
        checkpoint = torch.load(filepath)
        self.means = checkpoint['means']
        self.fitted = checkpoint['fitted']
        self.feature_dim = checkpoint.get('feature_dim', self.means.shape[1])
        self.flatten_tokens = checkpoint.get('flatten_tokens', False)


class OfflineCosineMuVarTracker:
    def __init__(self, num_domains, feature_dim=None, flatten_tokens=False):
        """
        Initialize offline domain statistics tracker using cosine similarity
        with both mean and variance information.
        
        Args:
            num_domains: Number of domains to track
            feature_dim: Feature dimension (optional, will be determined from data if None)
            flatten_tokens: Whether to flatten token-level features or average them
        """
        self.num_domains = num_domains
        self.feature_dim = feature_dim
        self.flatten_tokens = flatten_tokens
        self.means = None  # Will be initialized on first fit
        self.vars = None   # Will be initialized on first fit
        self.fitted = [False] * self.num_domains  # Flag

    def refit(self, features, domain_id):
        """
        Refit the mean (centroid) and variance from scratch.

        Args:
            features: All features for this domain, shape [batch_size, ...] or [batch_size, num_tokens, embed_dim]
            domain_id: Domain identifier.
        """
        with torch.no_grad():
            if not features.is_cuda:
                features = features.cuda()
            
            # Handle first fit - determine dimensions and initialize storage
            if self.means is None:
                if len(features.shape) == 3 and self.flatten_tokens:
                    # Handle token-level features with flattening
                    batch_size, num_tokens, embed_dim = features.shape
                    self.feature_dim = num_tokens * embed_dim
                    features = features.reshape(batch_size, -1)
                elif len(features.shape) == 3 and not self.flatten_tokens:
                    # Average token embeddings
                    batch_size, num_tokens, embed_dim = features.shape
                    self.feature_dim = embed_dim
                    features = features.mean(dim=1)
                else:
                    # Simple feature vector
                    self.feature_dim = features.shape[1]
                
                self.means = torch.zeros(self.num_domains, self.feature_dim).cuda()
                self.vars = torch.ones(self.num_domains, self.feature_dim).cuda()
            
            # Process features according to settings
            elif len(features.shape) == 3:
                if self.flatten_tokens:
                    # Flatten token dimension into feature dimension
                    batch_size, num_tokens, embed_dim = features.shape
                    features = features.reshape(batch_size, -1)
                else:
                    # Average token embeddings
                    features = features.mean(dim=1)
            
            # Calculate mean
            self.means[domain_id] = torch.mean(features, dim=0)
            # Calculate variance
            self.vars[domain_id] = torch.var(features, dim=0, unbiased=True)
            # Ensure numerical stability
            self.vars[domain_id] = torch.clamp(self.vars[domain_id], min=1e-5)
            self.fitted[domain_id] = True

    def get_domain_weights(self, features):
        """
        Compute domain similarity weights considering both mean and variance.
        Features are normalized by the domain's standard deviation before
        computing cosine similarity.
        """
        if self.means is None:
            raise RuntimeError("Tracker not fitted. Call refit() first.")
            
        if not features.is_cuda:
            features = features.cuda()
            
        # Process features according to settings
        if len(features.shape) == 3:
            if self.flatten_tokens:
                # Flatten token dimension into feature dimension
                batch_size, num_tokens, embed_dim = features.shape
                features = features.reshape(batch_size, -1)
            else:
                # Average token embeddings
                features = features.mean(dim=1)
                
        batch_size = features.size(0)
        similarities = torch.zeros(batch_size, self.num_domains).cuda()

        for domain_id in range(self.num_domains):
            if self.fitted[domain_id]:
                domain_std = torch.sqrt(self.vars[domain_id])

                # Normalize features by the standard deviation
                normalized_features = features / (domain_std + 1e-5)   

                # Normalize mean by the standard deviation           
                normalized_mean = self.means[domain_id] / (domain_std + 1e-5)           
                
                similarities[:, domain_id] = F.cosine_similarity(
                    normalized_features, 
                    normalized_mean.unsqueeze(0), 
                    dim=1
                )
            else:
                similarities[:, domain_id] = torch.zeros(batch_size).cuda()

        domain_weights = torch.softmax(similarities*2, dim=1)
        return domain_weights

    def save_cosine(self, filepath):
        """Save the tracker's parameters."""
        torch.save({
            'means': self.means,
            'vars': self.vars,
            'fitted': self.fitted,
            'feature_dim': self.feature_dim,
            'flatten_tokens': self.flatten_tokens
        }, filepath)

    def load_cosine(self, filepath):
        """Load the tracker's parameters."""
        checkpoint = torch.load(filepath)
        self.means = checkpoint['means']
        self.vars = checkpoint['vars']
        self.fitted = checkpoint.get('fitted', [True] * self.num_domains)
        self.feature_dim = checkpoint.get('feature_dim', self.means.shape[1])
        self.flatten_tokens = checkpoint.get('flatten_tokens', False)