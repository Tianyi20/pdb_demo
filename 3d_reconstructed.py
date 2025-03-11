import open3d as o3d
import numpy as np

def fill_objects(points, grid_resolution=100):
    """
    Given the surface point cloud of an object, fill in points between the surface and the ground plane (z=0).
    For each surface point, the function uses its local normal (computed via Open3D) to determine the straight-line 
    path from the point to the ground. Then grid_resolution number of points are linearly interpolated along that line.

    :param points: A NumPy array of shape (N, 3) representing the surface point cloud.
    :param grid_resolution: The number of interpolation steps (including the endpoints).
    :return: A NumPy array of shape (N * grid_resolution, 3) containing the filled point cloud.
    """
    # Create a point cloud and compute normals.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Estimate normals (adjust radius and max_nn if needed).
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(pcd.normals)

    filled_points_list = []
    for p, n in zip(points, normals):
        # Ensure the normal is upward oriented.
        if n[2] < 0:
            n = -n
        # If the normal's z-component is nearly zero, default to vertical fill.
        if abs(n[2]) < 1e-6:
            ground = np.array([p[0], p[1], 0])
        else:
            # Find t such that (p - t*n)[2] == 0, i.e. p_z - t*n_z = 0
            t = p[2] / n[2]
            ground = p - t * n
        
        # Interpolate linearly between the ground point and the surface point.
        line_points = np.linspace(ground, p, grid_resolution)
        filled_points_list.append(line_points)
    
    # Combine all filled points into one array.
    filled_points = np.concatenate(filled_points_list, axis=0)
    return filled_points

# Example usage:
if __name__ == "__main__":
    # Load surface point cloud from a PCD file (adjust the path as needed)
    pcd = o3d.io.read_point_cloud("tabletop_objects.pcd")
    points = np.asarray(pcd.points)
    
    # Fill the object interior points.
    filled = fill_objects(points, grid_resolution=50)
    
    # Visualize: combine the original surface and the filled interior.
    surface_pcd = o3d.geometry.PointCloud()
    surface_pcd.points = o3d.utility.Vector3dVector(points)
    filled_pcd = o3d.geometry.PointCloud()
    filled_pcd.points = o3d.utility.Vector3dVector(filled)
    
    # Create a coordinate frame and a small sphere at the origin for reference.
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    origin_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    origin_marker.paint_uniform_color([1, 0, 0])
    
    o3d.visualization.draw_geometries([surface_pcd, filled_pcd, coordinate_frame, origin_marker])
