import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import mahalanobis

class DomainMahalanobisTracker:
    def __init__(self, feature_dim, num_domains):
        """
        Initialize domain statistics tracker using Mahalanobis distance.

        Args:
            feature_dim: Dimensionality of the features.
            num_domains: Number of domains.
        """
        self.feature_dim = feature_dim
        self.num_domains = num_domains

        # Store means and inverse covariance matrices
        self.means = torch.zeros(self.num_domains, self.feature_dim).cuda()
        self.inv_covariances = [None] * self.num_domains  # Store as list, since they are matrices
        self.counts = torch.zeros(self.num_domains).cuda()
        self.fitted = [False] * num_domains

    def update(self, features, domain_id):
        """
        Update running statistics (mean and covariance) for a domain.

        Args:
            features: Batch of feature vectors (B, feature_dim) on CUDA.
            domain_id: Domain identifier.
        """
        with torch.no_grad():
            if not features.is_cuda:
                features = features.cuda()

            batch_size = features.size(0)
            new_count = self.counts[domain_id] + batch_size

            # Update mean (same as before)
            batch_mean = torch.mean(features, dim=0)
            delta = batch_mean - self.means[domain_id]
            self.means[domain_id] += delta * (batch_size / new_count)
            self.counts[domain_id] = new_count
            # Calculate the unbiased sample covariance matrix. Use .cpu() for numpy.
            if batch_size > 1:  # Need at least 2 samples to calculate covariance
                batch_cov = torch.cov(features.T)
            else:
                batch_cov = torch.zeros((self.feature_dim, self.feature_dim)).cuda()
            
            # Update the inverse covariance.
            self.inv_covariances[domain_id] = self._update_inv_covariance(batch_cov, domain_id)

            self.fitted[domain_id] = True # set the flag to true.
    
    def _update_inv_covariance(self, batch_cov, domain_id):
      """Update or compute the inverse covariance for a domain.
      Handles the case where the batch covariance might be singular."""
      # Regularization: Add a small value to the diagonal to prevent singularity
      reg_cov = 1e-6 * torch.eye(self.feature_dim).cuda()
      try:
          # Calculate the inverse using torch.linalg.inv
          inv_cov = torch.linalg.inv(batch_cov + reg_cov)
          return inv_cov

      except torch.linalg.LinAlgError:
          # Fallback if matrix is singular even after regularization
          print(f"Warning: Covariance matrix for domain {domain_id} is singular. Using pseudo-inverse.")
          # Use the pseudo-inverse as a fallback
          return torch.linalg.pinv(batch_cov + reg_cov)


    def get_domain_weights(self, features):
        """
        Compute domain similarity weights using (negative) Mahalanobis distance and softmax.

        Args:
            features: Input feature tensor of shape (batch_size, feature_dim).

        Returns:
            domain_weights: Normalized weights for each domain (batch_size, num_domains).
        """
        if not features.is_cuda:
            features = features.cuda()
        batch_size = features.size(0)
        distances = torch.zeros(batch_size, self.num_domains).cuda()

        for domain_id in range(self.num_domains):
            if self.fitted[domain_id]:  # Make sure stats have been calculated
              domain_mean = self.means[domain_id]
              inv_cov = self.inv_covariances[domain_id]
              for i in range(batch_size):
                  # Manually compute Mahalanobis distance using the formula
                  diff = features[i] - domain_mean
                  distances[i, domain_id] = torch.sqrt(torch.matmul(torch.matmul(diff.unsqueeze(0), inv_cov), diff.unsqueeze(1))) # unsqueeze to make sure it's 2d
            else:
              # Assign a default value if we have not calculated stats yet
              distances[i, domain_id] = 10000.0 # this is an inf distance

        # Convert to similarity weights:  Negative distance, then softmax.
        domain_weights = torch.softmax(-distances, dim=1)
        return domain_weights

    def save_mahalanobis(self, filepath):
      """Saves the tracker to a file."""
      torch.save({'means': self.means, 'inv_covariances': self.inv_covariances, 'counts':self.counts, 'fitted': self.fitted}, filepath)


    def load_mahalanobis(self, filepath):
      """Loads tracker's parameters from a file."""
      checkpoint = torch.load(filepath)
      self.means = checkpoint['means']
      self.inv_covariances = checkpoint['inv_covariances']
      self.counts = checkpoint['counts']
      self.fitted = checkpoint['fitted']