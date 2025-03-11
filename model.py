import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Reduce
from pytorch3d.ops import knn_points
from pytorch3d.loss import chamfer_distance
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR
import open3d as o3d
import numpy as np
from torch import distributions as dist

class ARPE(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, npoints=1024):
        super(ARPE, self).__init__()

        self.k = 32
        self.lin1 = nn.Linear(2 * in_channels, 2 * in_channels)
        self.lin2 = nn.Linear(2 * in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(2 * in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.max_pooling_layer = Reduce('bn k f -> bn 1 f', 'max')

    def forward(self, x):
        B, N, C = x.shape

        K = min(self.k, N)
        knn = knn_points(x, x, K=K, return_nn=True)[2]

        diffs = x.unsqueeze(2) - knn
        x = torch.cat([x.unsqueeze(2).repeat(1, 1, K, 1), diffs], dim=-1)

        x = F.elu(self.bn1(self.lin1(x.view(B * N, K, 2 * C)).transpose(1, 2)).transpose(1, 2))
        x = self.max_pooling_layer(x).squeeze(2)
        x = F.elu(self.bn2(self.lin2(x.view(B, N, 2 * C)).transpose(1, 2)).transpose(1, 2))
        return x
import torch.nn.utils.rnn as rnn_utils
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.sa = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x, mask=None):
        x = self.ln1(x)
        x = self.sa(x, x, x, attn_mask=mask)[0] + x
        x = self.ff(self.ln2(x)) + x
        return x

class PointCloudDecoder(nn.Module):
    def __init__(self, embed_dim, output_points):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, output_points * 3)
        )

    def forward(self, x):
        B, _, C = x.shape
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x.view(B, -1, 3)

class Adapt_gen(nn.Module):
    def __init__(self, cfg, embed_dim, n_points, n_blocks=3, groups=1):
        super().__init__()
        self.cfg = cfg
        self.arpe = ARPE(in_channels=3, out_channels=embed_dim, npoints=n_points)
        self.blocks = nn.ModuleList([TransformerBlock(d_model=embed_dim, n_heads=groups) for _ in range(n_blocks)])
        self.decoder = PointCloudDecoder(embed_dim, output_points=n_points // 2)


    def forward(self, x):
        device = next(self.parameters()).device

        if isinstance(x, list):
            x = [torch.tensor(pc, dtype=torch.float32, device=device) for pc in x]
            x = rnn_utils.pad_sequence(x, batch_first=True)

        x = self.arpe(x)
        for block in self.blocks:
            x = block(x)
        x = self.decoder(x)
        return x


def chamfer_loss(output_points, gt_points):
    """Compute Chamfer Distance Loss"""

    # Check input shapes
    assert len(output_points.shape) == 3, f"output_points has wrong shape {output_points.shape}"
    assert len(gt_points.shape) == 3, f"gt_points has wrong shape {gt_points.shape}"

    cd_loss, _ = chamfer_distance(output_points, gt_points)
    return cd_loss

def laplacian_smoothing_loss(points):
    """Encourages smooth local structures using K-Nearest Neighbors (KNN)."""
    knn = knn_points(points, points, K=5)
    neighbor_indices = knn[1]

    batch_size, num_points, _ = points.shape

    neighbor_indices = neighbor_indices.long()

    neighbors = torch.stack([
        points[b, neighbor_indices[b]] for b in range(batch_size)
    ], dim=0)

    laplacian_loss = torch.mean(torch.norm(points - torch.mean(neighbors, dim=2), dim=-1))

    return laplacian_loss

def edge_preserving_loss(output_points, gt_points):
    """Preserves edges in the reconstructed point cloud."""
    B, P, _ = output_points.shape
    G = gt_points.shape[1]

    if G < P:
        repeat_factor = P // G + 1
        gt_points = gt_points.repeat(1, repeat_factor, 1)[:, :P, :]

    diff = torch.abs(output_points - gt_points)
    edge_loss = torch.mean(F.relu(diff - 0.02))
    return edge_loss

def variance_regularization(output_points):
    """Encourage diversity in output points to prevent mode collapse."""
    mean_point = torch.mean(output_points, dim=1, keepdim=True)
    variance = torch.mean(torch.norm(output_points - mean_point, dim=-1))

    return torch.exp(-variance)
def repulsion_loss(points, k=5):
    """Encourage uniform point distribution by penalizing close neighbors."""
    B, P, _ = points.shape

    K = min(k, P - 1)
    if K < 1:
        return torch.tensor(0.0, device=points.device)

    knn = knn_points(points, points, K=K + 1)[2]
    if knn is None:
        return torch.tensor(0.0, device=points.device)

    neighbor_dists = torch.norm(knn[:, :, 1:, :] - knn[:, :, :-1, :], dim=-1)
    loss = torch.mean(torch.exp(-neighbor_dists))
    return loss

class Adapt_gen_pl(LightningModule):
    def __init__(self, cfg, embed_dim, n_points, n_blocks=3, groups=1):
        super().__init__()
        self.cfg = cfg
        self.model = Adapt_gen(cfg, embed_dim, n_points, n_blocks, groups)
        self.lr = cfg.train.lr
        self.weight_decay = cfg.train.weight_decay

        loss_values = torch.tensor(Adapt_gen_pl.load_loss_values("loss_convonet"), dtype=torch.float32)

        self.register_buffer("loss_convonet", loss_values)

    def load_loss_values(filepath):
        """Load loss values from a file"""
        with open(filepath, "r") as f:
            return [float(line.strip()) for line in f.readlines()]

    def compute_loss(self, input_points, gt_points):
        """Final combined loss function for training the point cloud autoencoder"""

        output_points = self.model(input_points)

        if isinstance(output_points, list):
            output_points = [torch.tensor(pc, dtype=torch.float32, device=self.device) for pc in output_points]
            output_points = torch.nn.utils.rnn.pad_sequence(output_points, batch_first=True)  # Pad variable length

        if isinstance(gt_points, list):
            gt_points = [torch.tensor(pc, dtype=torch.float32, device=self.device) for pc in gt_points]
            gt_points = torch.nn.utils.rnn.pad_sequence(gt_points, batch_first=True)

        output_points = output_points.view(output_points.shape[0], -1, 3)
        gt_points = gt_points.view(gt_points.shape[0], -1, 3)

        chamfer = chamfer_loss(output_points, gt_points)
        laplacian = laplacian_smoothing_loss(output_points)
        B, P, _ = output_points.shape
        gt_points = gt_points[:, :P, :]
        edge_reg = edge_preserving_loss(output_points, gt_points)
        #repulsion = repulsion_loss(output_points, k=5)

        loss = chamfer + 0.1 * laplacian+ 0.1 * variance_regularization(output_points)+edge_reg*0.1#+repulsion*0.1

        self.log('train_chamfer_loss', chamfer, prog_bar=True)
        self.log('train_laplacian_loss', laplacian, prog_bar=True)
        self.log('train_edge_loss', edge_reg, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        input_points, gt_points = batch
        loss = self.compute_loss(input_points, gt_points)
        if batch_idx < len(self.loss_convonet):
            loss += self.loss_convonet[batch_idx]*0.1
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_points, gt_points = batch
        loss = self.compute_loss(input_points, gt_points)
        if batch_idx < len(self.loss_convonet):
            loss += self.loss_convonet[batch_idx]*0.1
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=2 * self.cfg.train.epochs, eta_min=1e-6)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

