import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import mahalanobis

class OfflineMahalanobisTracker:
    def __init__(self, feature_dim, num_domains):
        """
        Initialize offline domain statistics tracker using Mahalanobis distance.
        """
        self.feature_dim = feature_dim
        self.num_domains = num_domains
        self.means = torch.zeros(self.num_domains, self.feature_dim).cuda()
        self.inv_covariances = [None] * self.num_domains
        self.fitted = [False] * self.num_domains

    def refit(self, features, domain_id):
        """
        Refit the mean and inverse covariance from scratch.
        """
        with torch.no_grad():
            if not features.is_cuda:
                features = features.cuda()

            # Calculate mean
            self.means[domain_id] = torch.mean(features, dim=0)

            # Calculate unbiased sample covariance
            if features.shape[0] > 1:
                batch_cov = torch.cov(features.T)
            else:
                batch_cov = torch.zeros((self.feature_dim, self.feature_dim)).cuda()
            # Regularization and inverse calculation (same as online version)
            self.inv_covariances[domain_id] = self._update_inv_covariance(batch_cov, domain_id)
            self.fitted[domain_id] = True


    def _update_inv_covariance(self, batch_cov, domain_id):
        """Update or compute the inverse covariance (same as online)."""
        reg_cov = 1e-6 * torch.eye(self.feature_dim).cuda()
        try:
            inv_cov = torch.linalg.inv(batch_cov + reg_cov)
            return inv_cov
        except torch.linalg.LinAlgError:
            print(f"Warning: Covariance matrix for domain {domain_id} is singular. Using pseudo-inverse.")
            return torch.linalg.pinv(batch_cov + reg_cov)

    def get_domain_weights(self, features):
        """
        Compute domain similarity weights (same as online version).
        """
        if not features.is_cuda:
            features = features.cuda()
        batch_size = features.size(0)
        distances = torch.zeros(batch_size, self.num_domains).cuda()

        for domain_id in range(self.num_domains):
          if self.fitted[domain_id]:
            domain_mean = self.means[domain_id]
            inv_cov = self.inv_covariances[domain_id]

            for i in range(batch_size):
                diff = features[i] - domain_mean
                distances[i, domain_id] = torch.sqrt(torch.matmul(torch.matmul(diff.unsqueeze(0), inv_cov), diff.unsqueeze(1)))
          else:
              distances[i, domain_id] = 10000.0

        domain_weights = torch.softmax(-distances, dim=1)
        return domain_weights

    def save_mahalanobis(self, filepath):
        """Save the tracker's parameters."""
        torch.save({'means': self.means, 'inv_covariances': self.inv_covariances, 'fitted': self.fitted}, filepath)

    def load_mahalanobis(self, filepath):
        """Load the tracker's parameters."""
        checkpoint = torch.load(filepath)
        self.means = checkpoint['means']
        self.inv_covariances = checkpoint['inv_covariances']
        self.fitted = checkpoint['fitted']