import glob
import os
from typing import List, Optional

import h5py
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import open3d as o3d

def random_point_dropout(pc, max_dropout_ratio=0.2):
    """Randomly drop points in the point cloud."""
    dropout_ratio = np.random.random() * max_dropout_ratio
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]
    return pc

def normalize_pointcloud(pc):
    pc = pc - np.mean(pc, axis=0)
    max_val = np.max(np.linalg.norm(pc, axis=1))
    pc = pc / max_val
    return pc

def translate_pointcloud(pointcloud):
    """Apply random scaling and translation to the point cloud."""
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

class ModelNet40Ply2048(Dataset):
    def __init__(self, root, split="train", train_ratio=1.0):
        self.ply_files = sorted(glob.glob(os.path.join(root, "*.ply")))
        assert len(self.ply_files) > 0, f"No .ply files found in {root}"

        split_index = int(len(self.ply_files) * train_ratio)

        if split == "train":
            self.ply_files = self.ply_files[:split_index]
        else:
            self.ply_files = self.ply_files[:split_index]


    def __getitem__(self, item):
        file_path = self.ply_files[item]
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points).astype(np.float32)

        #if points.shape[0] < 2048:
        #    points = np.random.rand(2048, 3).astype(np.float32)

        points = random_point_dropout(points)
        #points=normalize_pointcloud(points)
        #points = translate_pointcloud(points)
        #np.random.shuffle(points)

        return points, points

    def __len__(self):
        return len(self.ply_files)

import torch
def collate_fn(batch):
    """Custom collate function that ensures all point clouds are tensors."""
    pointclouds = [torch.tensor(item[0], dtype=torch.float32) for item in batch]
    targets = [torch.tensor(item[1], dtype=torch.float32) for item in batch]

    return pointclouds, targets

class ModelNet40Ply2048DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for ModelNet40Ply2048 with .ply files.
    """

    def __init__(
        self,
        data_dir: str = "/datasets/ply_shapenet",
        batch_size: int = 32,
        drop_last: bool = True,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        """Set up the datasets for training and validation."""
        self.modelnet_train = ModelNet40Ply2048(self.data_dir, split="train")
        self.modelnet_test = ModelNet40Ply2048(self.data_dir, split="test")

    def train_dataloader(self):
        return DataLoader(
            self.modelnet_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.modelnet_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        return 40


    @property
    def num_points(self) -> int:
        """Return the number of points in each point cloud."""
        return 2048



