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
    def __init__(self, input_dimensions, output_dimensions, propagate_dimensions=64, space_dim=4, k=16):
        super().__init__()
        self.k = k
        self.space = nn.Linear(input_dimensions, space_dim)   # learnable projection
        self.features = nn.Linear(input_dimensions, propagate_dimensions)
        self.mlp = nn.Sequential(
            nn.Linear(2*propagate_dimensions, output_dimensions),
            nn.ReLU(),
            nn.Linear(output_dimensions, output_dimensions)
        )

    def forward(self, x, mask):
        # x: (B, N, F)
        # mask: (B, N) bool
        x_feats = self.features(x)  # (B, N, output_dimensions)
        x_feats = x_feats * mask.unsqueeze(-1)
        coords = self.space(x)      # (B, N, space_dim)

        # Mask out padded hits before distance computation
        coords_masked = coords.clone()
        coords_masked[~mask] = float('inf')

        B, N, F_out = x_feats.shape # shape is batch size, num. hits, latent space representation

        # Finding the nearest neighbors in the space S
        knn_idx = knn(coords_masked, k=min(self.k, N))  # (B, N, k)

        # Message passing
        
        agg = torch.zeros_like(x_feats)

        # Loops over each event in the batch
        for b in range(B):
            # Loops over each hit in the event
            for n in range(N):
                # Skips hits that aren't actually hits 
                if not mask[b, n]:
                    continue
                neighbors = knn_idx[b, n]  # k neighbors
                neighbor_coords = coords[b, neighbors]
                neighbor_feats = x_feats[b, neighbors]  # (k, F_out)

                # Distance between node and neighbors in learned space
                d2 = torch.sum((coords[b, n] - neighbor_coords) ** 2, dim=-1)
                weights = torch.exp(-10 * d2)  # (k,)
                weighted_feats = neighbor_feats * weights.unsqueeze(-1)  # (k, F_out)
                weighted_mean = weighted_feats.mean(dim=0)  # (F_out,

                agg[b, n] = weighted_mean

        # Concatenate original features and aggregated neighbors
        combined = torch.cat([x_feats, agg], dim=-1)  # (B, N, 2*F_out)

        out = self.mlp(combined)
        return out


class GravNetModel(nn.Module):
    def __init__(self, hit_dim=3, global_dim=10, hidden_dim=64, num_classes=2, k=16):
        super().__init__()
        self.layer1 = GravNetLayer(hit_dim, hidden_dim, k=k)
        self.layer2 = GravNetLayer(hidden_dim, hidden_dim, k=k)
        self.global_fc = nn.Linear(global_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
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
