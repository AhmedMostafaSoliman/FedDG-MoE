import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import numpy as np

class OfflineGMMTracker:
    def __init__(self, args):
        """
        Initialize offline domain statistics tracker using Gaussian Mixture Models.

        Args:
            args: Arguments containing site_list for num_domains
        """
        self.num_domains = len(args.site_list)
        self.feature_dim = None
        self.num_components = args.num_components
        self.covariance_type = args.covariance_type

        self.gmms = None  # Will be initialized on first fit
        self.fitted = [False] * self.num_domains

    def refit(self, features, domain_id):
        """
        Refit the GMM from scratch using all provided features.

        Args:
            features: All features for this domain (from the current round's training data).
            domain_id: Domain identifier.
        """
        with torch.no_grad():
            if not features.is_cuda:
                features = features.cuda()
                
            # Handle first fit - determine dimensions and initialize GMMs
            if self.gmms is None:
                self.feature_dim = features.shape[1]
                self.gmms = [GaussianMixture(n_components=self.num_components,
                                            covariance_type=self.covariance_type,
                                            random_state=42)
                             for _ in range(self.num_domains)]
                
            features_np = features.cpu().numpy()

            # Fit the GMM using sklearn
            self.gmms[domain_id].fit(features_np)
            self.fitted[domain_id] = True

    def get_domain_weights(self, features):
        """
        Compute raw domain similarities based on GMM log probabilities.
        
        Args:
            features: Input features of shape [batch_size, feature_dim]
            
        Returns:
            Raw similarity scores without softmax normalization
        """
        if self.gmms is None:
            raise RuntimeError("Tracker not fitted. Call refit() first.")
            
        if not features.is_cuda:
            features = features.cuda()
        features_np = features.cpu().numpy()
        batch_size = features.shape[0]

        log_probs = np.zeros((batch_size, self.num_domains))

        for domain_id in range(self.num_domains):
            if self.fitted[domain_id]:
                log_probs[:, domain_id] = self.gmms[domain_id].score_samples(features_np)
            else:
                log_probs[:, domain_id] = np.ones(batch_size) * -np.inf

        # Return raw log probabilities - no softmax applied
        return torch.tensor(log_probs, dtype=torch.float32).cuda()