import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("pointclouds/03642806_82edd31783edc77018a5de3a5f9a5881_dec21.ply")
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.9, max_nn=1000)
)
pcd.orient_normals_consistent_tangent_plane(k=1000)

o3d.visualization.draw_geometries([pcd], point_show_normal=True)

distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)

distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)

radius=7*avg_dist
bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd,
    o3d.utility.DoubleVector([radius, radius*2])
)

bpa_mesh=bpa_mesh.simplify_quadric_decimation(100000)
bpa_mesh.filter_smooth_laplacian(number_of_iterations=5)
bpa_mesh.remove_degenerate_triangles()

o3d.visualization.draw_geometries([bpa_mesh])

o3d.io.write_triangle_mesh("bpa2_mesh.obj", bpa_mesh)

pcd = o3d.io.read_point_cloud("pointclouds/03642806_82edd31783edc77018a5de3a5f9a5881_dec21.ply")
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=1000)
)
pcd.orient_normals_consistent_tangent_plane(k=1000)

o3d.visualization.draw_geometries([pcd], point_show_normal=True)

distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)

distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)

radius=7*avg_dist
bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd,
    o3d.utility.DoubleVector([radius, radius*2])
)

bpa_mesh=bpa_mesh.simplify_quadric_decimation(100000)
bpa_mesh.filter_smooth_laplacian(number_of_iterations=5)
bpa_mesh.remove_degenerate_triangles()

o3d.visualization.draw_geometries([bpa_mesh])

o3d.io.write_triangle_mesh("bpa_mesh.obj", bpa_mesh)

mesh1 = o3d.io.read_triangle_mesh("bpa2_mesh.obj")
mesh2 = o3d.io.read_triangle_mesh("bpa_mesh.obj")

merged_mesh = mesh1 + mesh2
o3d.visualization.draw_geometries([merged_mesh])

o3d.io.write_triangle_mesh("merged.obj", merged_mesh)

