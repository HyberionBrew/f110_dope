import csv
import yaml
import numpy as np

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def read_csv(file_path):
    points = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            points.append([float(row[0]), float(row[1])])
    return points

def calculate_transformation(origin1, origin2, orientation1, orientation2):
    # Compute translation
    translation = np.array(origin2[:2]) - np.array(origin1[:2])

    # Compute rotation
    # Assuming the orientation is given in quaternion (x, y, z, w)
    # Convert quaternion to a rotation matrix or euler angles as needed
    # For simplicity, let's assume it's a 2D rotation and we only need the z-component
    rotation_angle = np.arctan2(orientation2[1], orientation2[0]) - np.arctan2(orientation1[1], orientation1[0])
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [np.sin(rotation_angle), np.cos(rotation_angle)]])

    return translation, rotation_matrix

def transform_points(points, translation, rotation_matrix):
    transformed_points = []
    for point in points:
        transformed_point = np.dot(rotation_matrix, np.array(point).T).T + translation
        transformed_points.append(transformed_point.tolist())
    return transformed_points

# Read the YAML files for both map frames
map_frame_1 = read_yaml('map_frame_1.yaml')
map_frame_2 = read_yaml('map_frame_2.yaml')

# Read the CSV file containing the points
points_in_map_frame_2 = read_csv('points.csv')

# Calculate the necessary transformation
translation, rotation_matrix = calculate_transformation(
    map_frame_1['origin'], map_frame_2['origin'],
    map_frame_1['origin_orientation'], map_frame_2['origin_orientation']
)

# Apply the transformation
transformed_points = transform_points(points_in_map_frame_2, translation, rotation_matrix)

# Now transformed_points contains the points in the first map frame
print(transformed_points)