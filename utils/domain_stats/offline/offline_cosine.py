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
    

class OfflineCosineMuVarTracker:
    def __init__(self, args):
        """
        Initialize offline domain statistics tracker using cosine similarity
        with both mean and variance information.

        We divide the mean by the variance to down-weight features that are not stable within a domain
        
        Args:
            args: Arguments containing site_list for num_domains and normalize_features option
        """
        self.num_domains = len(args.site_list)
        self.feature_dim = None  # Will be determined from first batch
        self.means = None  # Will be initialized on first fit
        self.vars = None   # Will be initialized on first fit
        self.fitted = [False] * self.num_domains  # Flag
        # Whether to normalize features by standard deviation or use them as-is
        self.normalize_features = getattr(args, 'normalize_features', True)

    def refit(self, features, domain_id):
        """
        Refit the mean (centroid) and variance from scratch.
        Assumes features are already in shape [batch_size, feature_dim]

        Args:
            features: All features for this domain, shape [batch_size, feature_dim]
            domain_id: Domain identifier.
        """
        with torch.no_grad():
            if not features.is_cuda:
                features = features.cuda()
            
            # First fit - determine dimensions and initialize storage
            if self.means is None:
                self.feature_dim = features.shape[1]
                self.means = torch.zeros(self.num_domains, self.feature_dim).cuda()
                self.vars = torch.ones(self.num_domains, self.feature_dim).cuda()
            
            # Calculate mean
            self.means[domain_id] = torch.mean(features, dim=0)
            # Calculate variance
            self.vars[domain_id] = torch.var(features, dim=0, unbiased=True)
            # Ensure numerical stability
            self.vars[domain_id] = torch.clamp(self.vars[domain_id], min=1e-5)
            self.fitted[domain_id] = True

    def get_domain_weights(self, features):
        """
        Compute raw domain similarities considering both mean and variance.
        Features are normalized by the domain's standard deviation before
        computing cosine similarity if normalize_features is True.
        
        Args:
            features: Input features of shape [batch_size, feature_dim]
            
        Returns:
            Raw similarity scores without softmax normalization
        """
        if self.means is None:
            raise RuntimeError("Tracker not fitted. Call refit() first.")
            
        if not features.is_cuda:
            features = features.cuda()
                
        batch_size = features.size(0)
        similarities = torch.zeros(batch_size, self.num_domains).cuda()

        for domain_id in range(self.num_domains):
            if self.fitted[domain_id]:
                domain_std = torch.sqrt(self.vars[domain_id])

                # Normalize features by the standard deviation if enabled
                if self.normalize_features:
                    normalized_features = features / (domain_std + 1e-5)
                else:
                    normalized_features = features   

                # Normalize mean by the standard deviation           
                normalized_mean = self.means[domain_id] / (domain_std + 1e-5)           
                
                similarities[:, domain_id] = F.cosine_similarity(
                    normalized_features, 
                    normalized_mean.unsqueeze(0), 
                    dim=1
                )
            else:
                similarities[:, domain_id] = torch.zeros(batch_size).cuda()

        # Return raw similarities - no softmax applied
        return similarities

class OfflineCosineConcat:
    def __init__(self, args):
        """
        Initialize offline domain statistics tracker that concatenates 
        mean and variance vectors before computing cosine similarity.
        
        Args:
            args: Arguments containing site_list for num_domains
        """
        self.num_domains = len(args.site_list)
        self.feature_dim = None  # Will be determined from first batch
        self.means = None  # Will be initialized on first fit
        self.vars = None   # Will be initialized on first fit
        self.fitted = [False] * self.num_domains  # Flag

    def refit(self, features, domain_id):
        """
        Refit the mean (centroid) and variance from scratch.
        Assumes features are already in shape [batch_size, feature_dim]

        Args:
            features: All features for this domain, shape [batch_size, feature_dim]
            domain_id: Domain identifier.
        """
        with torch.no_grad():
            if not features.is_cuda:
                features = features.cuda()
            
            # First fit - determine dimensions and initialize storage
            if self.means is None:
                self.feature_dim = features.shape[1]
                self.means = torch.zeros(self.num_domains, self.feature_dim).cuda()
                self.vars = torch.ones(self.num_domains, self.feature_dim).cuda()
            
            # Calculate mean
            self.means[domain_id] = torch.mean(features, dim=0)
            # Calculate variance
            self.vars[domain_id] = torch.var(features, dim=0, unbiased=True)
            # Ensure numerical stability
            self.vars[domain_id] = torch.clamp(self.vars[domain_id], min=1e-5)
            self.fitted[domain_id] = True

    def get_domain_weights(self, features):
        """
        Compute raw domain similarities by concatenating mean and variance vectors.
        First computes mean and variance of the input features (batch),
        then compares them with domain statistics.
        
        Args:
            features: Input features of shape [batch_size, feature_dim]
            
        Returns:
            Raw similarity scores without softmax normalization
        """
        if self.means is None:
            raise RuntimeError("Tracker not fitted. Call refit() first.")
            
        if not features.is_cuda:
            features = features.cuda()
                
        batch_size = features.size(0)
        
        # Compute mean and variance of input features batch
        feature_mean = torch.mean(features, dim=0)
        feature_var = torch.var(features, dim=0, unbiased=True)
        feature_var = torch.clamp(feature_var, min=1e-5)
        
        # Concatenate mean and variance of input features
        feature_concat = torch.cat([feature_mean, feature_var])
        
        # Initialize similarities tensor
        similarities = torch.zeros(batch_size, self.num_domains).cuda()
        
        for domain_id in range(self.num_domains):
            if self.fitted[domain_id]:
                # Concatenate mean and variance of domain
                domain_concat = torch.cat([self.means[domain_id], self.vars[domain_id]])
                
                # Compute cosine similarity between concatenated vectors - just once
                sim_value = F.cosine_similarity(
                    feature_concat.unsqueeze(0),
                    domain_concat.unsqueeze(0),
                    dim=1
                )
                
                # Assign the same similarity to all samples in the batch
                similarities[:, domain_id] = sim_value.item() * torch.ones(batch_size).cuda()
            else:
                similarities[:, domain_id] = torch.zeros(batch_size).cuda()

        # Return raw similarities - no softmax applied
        return similarities