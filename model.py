import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_distance(x):
    # x: (B, N, F)
    # returns (B, N, N) squared distances
    x_expand = x.unsqueeze(2)  # (B, N, 1, F)
    x_t_expand = x.unsqueeze(1)  # (B, 1, N, F)
    dist = torch.sum((x_expand - x_t_expand) ** 2, dim=-1)
    return dist


def knn(x, k=16):
    # x: (B, N, F)
    # returns indices of k nearest neighbors
    dist = pairwise_distance(x)
    knn_idx = dist.topk(k=k, dim=-1, largest=False).indices  # (B, N, k)
    return knn_idx


class GravNetLayer(nn.Module):
    def __init__(self, input_dimensions, output_dimensions, propagate_dimensions=64, space_dim=4, k=16, dropout_rate=0):
        super().__init__()
        self.k = k
        self.space = nn.Linear(input_dimensions, space_dim)   # learnable projection
        self.features = nn.Linear(input_dimensions, propagate_dimensions)
        self.mlp = nn.Sequential(
            nn.Linear(2*propagate_dimensions, output_dimensions),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dimensions, output_dimensions)
        )

    def forward(self, x, mask):
        # x: (B, N, F)
        # mask: (B, N) bool
        x_feats = self.features(x)  # (B, N, F_out)
        x_feats = x_feats * mask.unsqueeze(-1)
        coords = self.space(x)      # (B, N, space_dim)

        # Mask out padded hits before distance computation
        coords_masked = coords.clone()
        coords_masked[~mask] = float('inf')

        B, N, F_out = x_feats.shape
        k = min(self.k, N)

        # Get nearest neighbors (B, N, k)
        knn_idx = knn(coords_masked, k=k)

        # Build batch indices for gather
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1)
        batch_idx = batch_idx.expand(-1, N, k)

        # Gather neighbor features and coordinates
        neighbor_coords = coords[batch_idx, knn_idx]   # (B, N, k, space_dim)
        neighbor_feats  = x_feats[batch_idx, knn_idx]  # (B, N, k, F_out)

        # Compute distances to neighbors
        d2 = torch.sum((coords.unsqueeze(2) - neighbor_coords) ** 2, dim=-1)  # (B, N, k)

        # Compute weights
        weights = torch.exp(-10 * d2)  # (B, N, k)

        # Weighted mean of neighbor features
        weighted_sum = torch.sum(neighbor_feats * weights.unsqueeze(-1), dim=2)
        weight_norm = torch.sum(weights, dim=2, keepdim=True).clamp_min(1e-8)
        weighted_mean = weighted_sum / weight_norm  # (B, N, F_out)

        # Concatenate and pass through MLP
        combined = torch.cat([x_feats, weighted_mean], dim=-1)
        out = self.mlp(combined)
        return out

class GravNetModel(nn.Module):
    def __init__(self, hit_dim=3, global_dim=10, hidden_dim=64, num_classes=2, k=16, dropout_rate=0):
        super().__init__()
        self.layer1 = GravNetLayer(hit_dim, hidden_dim, k=k, dropout_rate=dropout_rate)
        self.layer2 = GravNetLayer(hidden_dim, hidden_dim, k=k, dropout_rate=dropout_rate)
        self.global_fc = nn.Linear(global_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, hits_padded, globals_event, mask):
        x = hits_padded
        x = self.layer1(x, mask)
        x = self.layer2(x, mask)

        # Aggregate per-event node features
        # Mask padded hits before mean
        x = x * mask.unsqueeze(-1)  # (B, N, F)
        hit_repr = x.sum(dim=1) / mask.sum(dim=1, keepdim=True)  # (B, F)

        # Global features
        global_repr = F.relu(self.global_fc(globals_event))
        # Concatenate and classify
        out = self.classifier(torch.cat([hit_repr, global_repr], dim=1))
        return out
