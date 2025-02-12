import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import numpy as np

class OnlineGMMTracker:
    def __init__(self, feature_dim, num_domains, num_components=5, covariance_type='full', max_iter=100, tol=1e-3, reg_covar=1e-6):
        """
        Initialize domain statistics tracker using Gaussian Mixture Models.

        Args:
            feature_dim: Dimensionality of the features.
            num_domains: Number of domains.
            num_components: Number of GMM components per domain.  This is a hyperparameter.
            covariance_type: Type of covariance matrix ('full', 'tied', 'diag', 'spherical').
            max_iter: Maximum number of EM iterations.
            tol: Convergence threshold for EM.
            reg_covar: Regularization added to the diagonal of covariance matrices.
        """
        self.feature_dim = feature_dim
        self.num_domains = num_domains
        self.num_components = num_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar

        self.gmms = [GaussianMixture(n_components=self.num_components,
                                    covariance_type=self.covariance_type,
                                    max_iter=self.max_iter,
                                    tol=self.tol,
                                    reg_covar=self.reg_covar,
                                    random_state=42)  # for reproducibility
                     for _ in range(self.num_domains)]
        self.fitted = [False] * num_domains # Flag to indicate if the GMM is fitted

    def update(self, features, domain_id):
        """
        Update GMM for a domain using provided features.  Uses sklearn's GMM.

        Args:
            features: Batch of feature vectors (B, feature_dim).
            domain_id: Domain identifier.
        """
        with torch.no_grad():
            if not features.is_cuda:
                features = features.cuda()
            features_np = features.cpu().numpy()

        # Fit the GMM using sklearn
        self.gmms[domain_id].fit(features_np)
        self.fitted[domain_id] = True

    def get_domain_weights(self, features):
        """
        Compute domain similarity weights using GMM log-likelihood with softmax.

        Args:
            features: Input feature tensor of shape (batch_size, feature_dim).

        Returns:
            domain_weights: Normalized weights for each domain (batch_size, num_domains).
        """
        if not features.is_cuda:
            features = features.cuda()
        features_np = features.cpu().numpy()
        batch_size = features.shape[0]

        log_probs = np.zeros((batch_size, self.num_domains))

        for domain_id in range(self.num_domains):
            if self.fitted[domain_id]:  # Check if GMM for this domain has been fitted
              log_probs[:, domain_id] = self.gmms[domain_id].score_samples(features_np)
            else: # if not return ones to avoid errors
              log_probs[:, domain_id] = np.ones(batch_size)

        # Convert to PyTorch tensor and apply softmax
        log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32).cuda()
        domain_weights = torch.softmax(log_probs_tensor, dim=1)
        return domain_weights

    def save_gmms(self, filepath):
      """Saves the fitted GMMs to a file."""
      torch.save({'gmms': self.gmms, 'fitted': self.fitted}, filepath)


    def load_gmms(self, filepath):
      """Loads GMMs from a file."""
      checkpoint = torch.load(filepath)
      self.gmms = checkpoint['gmms']
      self.fitted = checkpoint['fitted']