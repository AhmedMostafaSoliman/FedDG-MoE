import torch
import torch.nn as nn
import torch.nn.functional as F

class OnlineCosineTracker:
    def __init__(self, feature_dim, num_domains):
        """
        Initialize domain statistics tracker using cosine similarity to centroids.

        Args:
            feature_dim: Dimensionality of the features.
            num_domains: Number of domains.
        """
        self.feature_dim = feature_dim
        self.num_domains = num_domains

        # Store only the means (centroids)
        self.means = torch.zeros(self.num_domains, self.feature_dim).cuda()
        self.counts = torch.zeros(self.num_domains).cuda()
        self.fitted = [False] * self.num_domains  # Flag

    def update(self, features, domain_id):
        """
        Update running mean for a domain.

        Args:
            features: Batch of feature vectors (B, feature_dim) on CUDA.
            domain_id: Domain identifier.
        """
        with torch.no_grad():
            if not features.is_cuda:
                features = features.cuda()

            batch_size = features.size(0)
            new_count = self.counts[domain_id] + batch_size

            # Update mean (same as in previous trackers)
            batch_mean = torch.mean(features, dim=0)
            delta = batch_mean - self.means[domain_id]
            self.means[domain_id] += delta * (batch_size / new_count)
            self.counts[domain_id] = new_count
            self.fitted[domain_id] = True

    def get_domain_weights(self, features):
        """
        Compute domain similarity weights using cosine similarity to centroids and softmax.

        Args:
            features: Input feature tensor of shape (batch_size, feature_dim).

        Returns:
            domain_weights: Normalized weights for each domain (batch_size, num_domains).
        """
        if not features.is_cuda:
            features = features.cuda()
        batch_size = features.size(0)
        similarities = torch.zeros(batch_size, self.num_domains).cuda()

        for domain_id in range(self.num_domains):
          if self.fitted[domain_id]:
            domain_mean = self.means[domain_id]
            # Use F.cosine_similarity to calculate cosine similarity
            similarities[:, domain_id] = F.cosine_similarity(features, domain_mean.unsqueeze(0), dim=1)
          else:
            similarities[:, domain_id] = torch.zeros(batch_size).cuda()

        # Apply softmax to get normalized weights
        domain_weights = torch.softmax(similarities, dim=1)
        return domain_weights

    def save_cosine(self, filepath):
        """Save the tracker's parameters."""
        torch.save({'means': self.means, 'counts': self.counts, 'fitted': self.fitted}, filepath)

    def load_cosine(self, filepath):
        """Load the tracker's parameters."""
        checkpoint = torch.load(filepath)
        self.means = checkpoint['means']
        self.counts = checkpoint['counts']
        self.fitted = checkpoint['fitted']