import pyvista as pv
import open3d as o3d
import numpy as np

# Load the point cloud from a .ply file
point_cloud_file = "/export/livia/home/vision/Abahri/projects/SMART_PC/smart_pc/Figures/point_cloud/point_cloud_16.ply"  # Replace with your file path
skeleton_file = "/export/livia/home/vision/Abahri/projects/SMART_PC/smart_pc/Figures/skeleton/skeleton_16.ply"  # Replace with your file path

# Read point cloud
pcd = o3d.io.read_point_cloud(point_cloud_file)
point_cloud = pv.PolyData(np.asarray(pcd.points))

# Read skeleton points
skel = o3d.io.read_point_cloud(skeleton_file)
skeleton_points = pv.PolyData(np.asarray(skel.points))

# Create a PyVista plotter
plotter = pv.Plotter(off_screen=True)  # Enable off-screen rendering

# Add the point cloud with blue color
plotter.add_mesh(point_cloud, color="blue", point_size=5, render_points_as_spheres=True, label="Point Cloud")

# Add the skeleton points with red color
plotter.add_mesh(skeleton_points, color="red", point_size=10, render_points_as_spheres=True, label="Skeleton Points")

# Set the camera position for a better view (adjust as needed)
plotter.camera_position = "xy"

# Save the rendered image
output_file = "rendered_output.png"  # Specify the output file name
plotter.add_legend()
plotter.show_grid()
plotter.screenshot(output_file)

print(f"Rendered image saved to {output_file}")
