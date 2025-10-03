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
    def __init__(self, in_features, out_features, space_dim=4, k=16):
        super().__init__()
        self.k = k
        self.space = nn.Linear(in_features, space_dim)   # learnable projection
        self.features = nn.Linear(in_features, out_features)
        self.mlp = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )

    def forward(self, x, mask):
        # x: (B, N, F)
        # mask: (B, N) bool
        x_feats = self.features(x)  # (B, N, out_features)
        coords = self.space(x)      # (B, N, space_dim)

        # Mask out padded hits before distance computation
        coords_masked = coords.clone()
        coords_masked[~mask] = float('inf')

        B, N, F_out = x_feats.shape

        knn_idx = knn(coords_masked, k=min(self.k, N))  # (B, N, k)

        # Message passing
        
        agg = torch.zeros_like(x_feats)

        for b in range(B):
            for n in range(N):
                if not mask[b, n]:
                    continue
                neighbors = knn_idx[b, n]  # k neighbors
                neighbor_feats = x_feats[b, neighbors]  # (k, F_out)
                agg[b, n] = neighbor_feats.mean(dim=0)

        # Combine original features + aggregated neighbors
        out = self.mlp(x_feats + agg)
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
        global_event_flat = globals_event.squeeze(1)
        global_repr = F.relu(self.global_fc(global_event_flat))

        # Concatenate and classify
        out = self.classifier(torch.cat([hit_repr, global_repr], dim=1))
        return out
