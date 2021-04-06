import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("/home/ros2-foxy/uneven_2.pcd")

aabb = pcd.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)

box_corners = np.asarray(aabb.get_box_points())

min_corner = box_corners[0]
max_corner = box_corners[4]

print(box_corners)

x_dist = abs(max_corner[0] - min_corner[0])
y_dist = abs(max_corner[1] - min_corner[1])

step_size = 20

geometries = []

#geometries.append(aabb)

for x in range(0, int(x_dist / step_size + 1)):

    for y in range(0, int(y_dist / step_size + 1)):

        current_min_corner = [
            min_corner[0] + step_size * x,
            min_corner[1] + step_size * y,
            min_corner[2],
        ]

        current_max_corner = [
            current_min_corner[0] + step_size,
            current_min_corner[1] + step_size,
            max_corner[2],
        ]

        this_box = o3d.geometry.AxisAlignedBoundingBox(
            current_min_corner, current_max_corner
        )

        cropped_pcd = pcd.crop(this_box)
        print(cropped_pcd)
        geometries.append(cropped_pcd)
        geometries.append(this_box)
        if not cropped_pcd.has_points():
            print("PCL with no points !")

o3d.visualization.draw_geometries(geometries)
