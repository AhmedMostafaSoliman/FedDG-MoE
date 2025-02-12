import torch
import torch.nn as nn
import torch.nn.functional as F

class OfflineCosineTracker:
    def __init__(self, feature_dim, num_domains):
        """
        Initialize offline domain statistics tracker using cosine similarity.
        """
        self.feature_dim = feature_dim
        self.num_domains = num_domains
        self.means = torch.zeros(self.num_domains, self.feature_dim).cuda()
        self.fitted = [False] * self.num_domains  # Flag

    def refit(self, features, domain_id):
        """
        Refit the mean (centroid) from scratch.

        Args:
            features: All features for this domain.
            domain_id: Domain identifier.
        """
        with torch.no_grad():
            if not features.is_cuda:
                features = features.cuda()
            self.means[domain_id] = torch.mean(features, dim=0)
            self.fitted[domain_id] = True

    def get_domain_weights(self, features):
        """
        Compute domain similarity weights (same as online version).
        """
        if not features.is_cuda:
            features = features.cuda()
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
        torch.save({'means': self.means, 'fitted': self.fitted}, filepath)

    def load_cosine(self, filepath):
        """Load the tracker's parameters."""
        checkpoint = torch.load(filepath)
        self.means = checkpoint['means']
        self.fitted = checkpoint['fitted']