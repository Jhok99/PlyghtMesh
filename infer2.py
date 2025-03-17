import torch
import omegaconf
from pathlib import Path
import trimesh
import numpy as np
from pytorch_lightning import seed_everything
from metrics import *
from metrics_new import *
from model import Adapt_gen_pl
import random
from mesh_generation import *
import open3d as o3d
from metrics import *
import time
import re
def load_model_from_checkpoint(cfg, checkpoint_path):
    """Load the Adapt_gen_pl model from a checkpoint."""
    model = Adapt_gen_pl(
        cfg,
        embed_dim=cfg.model.embed_dim,
        n_points=cfg.n_points,
        n_blocks=cfg.model.n_blocks,
        groups=cfg.model.groups,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_pointclouds_from_directory(directory):
    """Load point cloud file paths from a directory."""
    ply_files = list(Path(directory).glob("*.ply"))
    if not ply_files:
        raise ValueError(f"No .ply files found in directory: {directory}")
    return ply_files


def load_pointcloud(file_path):
    """Load a point cloud from a .ply file."""
    pcd = trimesh.load(file_path)
    points = np.asarray(pcd.vertices)
    return points
import scipy.spatial



def remove_farthest_point(pcd):
    """
    Removes only the single farthest point from the centroid of the point cloud.

    Parameters:
    - pcd: open3d.geometry.PointCloud object

    Returns:
    - filtered_pcd: open3d.geometry.PointCloud object with the farthest point removed
    """
    points = np.asarray(pcd.points)

    if points.shape[0] == 0:
        print("Point cloud is empty!")
        return pcd

    centroid = np.mean(points, axis=0)

    distances = np.linalg.norm(points - centroid, axis=1)

    farthest_idx = np.argmin(distances)

    filtered_points = np.delete(points, farthest_idx, axis=0)

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    return filtered_pcd

from sklearn.neighbors import NearestNeighbors
def compute_normals_pca(vertices, k=20):
    """
    Compute normals for a point cloud using PCA on the local neighborhood.

    Parameters:
        vertices (np.ndarray): Array of shape (N, 3) with point coordinates.
        k (int): Number of nearest neighbors to consider.

    Returns:
        np.ndarray: Array of computed normals of shape (N, 3).
    """
    # Build nearest neighbor model
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(vertices)
    distances, indices = nbrs.kneighbors(vertices)

    normals = []
    for i in range(len(vertices)):
        neighbors = vertices[indices[i][1:]]
        centered = neighbors - np.mean(neighbors, axis=0)
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        normals.append(normal)
    return np.array(normals)


def compute_fixed_voxel_occupancy(points, grid_shape=(8, 8, 16)):
    """
    Computes an occupancy grid with a fixed number of voxels.
    The grid dimensions are fixed so that the total number of voxels is
    the product of the grid_shape dimensions (e.g., 8 x 8 x 16 = 1024).

    Args:
        points (numpy.ndarray): Array of shape (N, 3) representing the point cloud.
        grid_shape (tuple): Desired grid dimensions (e.g., (8, 8, 16) for 1024 cells).

    Returns:
        occupancy_flat (numpy.ndarray): A flattened 1D boolean array of occupancy with size equal to the product of grid_shape.
        occupancy_grid (numpy.ndarray): The 3D occupancy grid.
    """
    grid_origin = points.min(axis=0)
    grid_max = points.max(axis=0)
    bbox_size = grid_max - grid_origin

    normalized_points = (points - grid_origin) / bbox_size

    indices = np.floor(normalized_points * np.array(grid_shape)).astype(int)
    indices = np.clip(indices, 0, np.array(grid_shape) - 1)

    occupancy_grid = np.zeros(grid_shape, dtype=bool)
    occupancy_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    occupancy_flat = occupancy_grid.flatten()

    return occupancy_flat

@torch.no_grad()
def infer(cfg, checkpoint_path, num_samples=10, pointcloud_dir="/datasets/shapenet", output_dir="pointclouds"):
    """Perform inference and save generated meshes."""
    seed_everything(cfg.experiment.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(cfg, checkpoint_path).to(device)

    pointcloud_files = load_pointclouds_from_directory(pointcloud_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.shuffle(pointcloud_files)
    total_time=0.0
    num_generated= 0

    for i in range(min(num_samples, len(pointcloud_files))):
        pointcloud_file = pointcloud_files[i]

        points = load_pointcloud(pointcloud_file)
        points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(device)
        start_time = time.time()
        generated_points = model(points_tensor)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.5,
            max_nn=30
        ))
        original_normals=np.array(pcd.normals)
        if isinstance(generated_points, tuple):
            generated_points = generated_points[0]

        generated_points = generated_points[0].detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=1).fit(points)
        distances, indices = nbrs.kneighbors(generated_points)
        # Create Point Cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(generated_points)
        # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        #     radius=0.5,
        #     max_nn=30
        # ))
        modified_ply_path = output_dir / f"{pointcloud_file.stem}.ply"
        o3d.io.write_point_cloud(str(modified_ply_path), pcd, write_ascii=True)

        print(f"Modified point cloud saved: {modified_ply_path}")

        # # Ensure the point cloud has points before processing
        # if len(pcd.points) == 0:
        #     raise ValueError("Error: Point cloud is empty after model generation.")
        #
        #
        # camera_location = [3.0, 3.0, 5.0]  # for example
        # pcd.orient_normals_towards_camera_location(camera_location)
        # normals = original_normals[indices[:,0]]
        # voxel_size = 0.1
        # occupancy_grid= compute_fixed_voxel_occupancy(generated_points,grid_shape=(8, 8, 16))
        #
        # # Compute bounding box location and scale
        # bbox = np.array([generated_points.min(axis=0), generated_points.max(axis=0)])
        # loc = (bbox[0] + bbox[1]) / 2
        # scale = (bbox[1] - bbox[0]).max()
        # match = re.match(r"(\d+)_(\w+)_dec\d+", pointcloud_file.stem)
        # if not match:
        #     raise ValueError(f"Filename {pointcloud_file.stem} does not match expected pattern!")
        #
        # category, identifier = match.groups()
        #
        # # Define new directory structure
        # category_dir = output_dir / category
        # identifier_dir = category_dir / identifier
        #
        # # Ensure directories exist
        # identifier_dir.mkdir(parents=True, exist_ok=True)
        #
        # # Define NPZ file path inside the structured directories
        # npz_path = identifier_dir / "pointcloud.npz"
        #
        # # Save the point cloud data as NPZ
        # np.savez(npz_path, points=generated_points, normals=normals, loc=loc, scale=scale)
        #
        # npz_path= identifier_dir / "points.npz"
        # np.savez(npz_path, points=generated_points, occupancies=occupancy_grid, loc=loc, scale=scale)
        #
        #
        # print(f"Saved generated point cloud: {npz_path}")

        # try:
        # pcd = trimesh.load(modified_ply_path)
        # pointcloud = np.asarray(pcd.vertices)
        #
        # resolution, voxel_density, radius, splat_radius, threshold = compute_adaptive_parameters(pointcloud)
        #
        # grid, min_bounds, max_bounds = points_to_volume(
        #     pointcloud, resolution=resolution, voxel_density=voxel_density, radius=radius, splat_radius=splat_radius
        # )
        #
        # mesh = volume_to_mesh(grid, min_bounds, max_bounds, resolution=resolution, threshold=threshold)

        #
        #
        #
        #     end_time = time.time()
        #     generation_time = end_time - start_time
        #     total_time += generation_time
        #     num_generated += 1
        #
        #
        # mesh_path = f"{output_dir}/{pointcloud_file.stem}.obj"
        # save_mesh_as_obj(mesh, mesh_path)
        # except Exception as e:
        #     print(f"Failed to create mesh for {pointcloud_file.name}: {e}")

    # avg_time= total_time/num_generated if num_generated >0 else 0.0
    # print(f"\nAverage Mesh Generation Time : {avg_time:.6f} seconds")





#python infer2.py config.yaml checkpoints/norepulsion-convonet-lowmasknotransl-epoch=20-val_loss=1.02990.ckpt 10 datasets/ply_modelnet40
#python infer2.py config.yaml checkpoints/convonet-noedgeloss-lowmasknotransl-epoch=16-val_loss=0.99170.ckpt 10 datasets/ply_modelnet40



#python infer2.py config.yaml checkpoints/convonet-lowmasknotransl-epoch=17-val_loss=1.02990.ckpt 10 datasets/ply_modelnet40
#python infer2.py config.yaml checkpoints/convonet-lowmasktransl-epoch=11-val_loss=1.03599.ckpt 10 datasets/ply_modelnet40

#python infer2.py config.yaml checkpoints/convonet-highmasktransl-epoch=17-val_loss=1.03803.ckpt 10 datasets/ply_modelnet40
#python infer2.py config.yaml checkpoints/convonet-highmasknotransl-epoch=11-val_loss=1.03287.ckpt 10 datasets/ply_modelnet40





#python infer2.py config.yaml checkpoints/newloss-avoid-center-epoch=40-val_loss=0.04736.ckpt 10 datasets/ply_modelnet40


#python infer2.py config.yaml checkpoints/lesspoint-lowmasknotransl-epoch=48-val_loss=0.07462.ckpt 10 datasets/ply_modelnet40
#python infer2.py config.yaml checkpoints/shapenet-lowmasknotransl-epoch=04-val_loss=0.05294.ckpt 10 datasets/ply_modelnet40

#LOWMASKNOTRANSL
#python infer2.py config.yaml checkpoints/norep-lesspoint-lowmasknotransl-epoch=182-val_loss=0.04914.ckpt 10 datasets/outline_ply

#LOWMASKTRANSL
#python infer2.py config.yaml checkpoints/norep-lesspoint-lowmasktransl-epoch=81-val_loss=0.05653.ckpt 10 datasets/outline_ply

#HIGHMASKNOTRANSL
#python infer2.py config.yaml checkpoints/norep-lesspoint-highmasknotransl-epoch=164-val_loss=0.05032.ckpt 10 datasets/outline_ply

#HIGHMASKTRANSL
#python infer2.py config.yaml checkpoints/norep-lesspoint-highmasktransl-epoch=75-val_loss=0.05672.ckpt 10 datasets/outline_ply

#python infer2.py config.yaml checkpoints/lowmasknotransl-epoch=33-val_loss=0.06923.ckpt 10 datasets/outline_ply



if __name__ == "__main__":
    import sys

    if len(sys.argv) != 5:
        print("Usage: python infer.py <config_path> <checkpoint_path> <num_samples> <pointcloud_dir>")
        sys.exit(1)

    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    num_samples = int(sys.argv[3])
    pointcloud_dir = sys.argv[4]

    cfg = omegaconf.OmegaConf.load(config_path)

    infer(cfg, checkpoint_path, num_samples=num_samples, pointcloud_dir=pointcloud_dir)



