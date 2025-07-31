import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import f_oneway  # For ANOVA test
import re

# Load the CSV file
file_path = '/Users/ivy/Desktop/spot_gesture_eval/1003/consistency_test/consistency_test.csv'
df = pd.read_csv(file_path)
df = df.drop(index=6)  # Remove row at index 6

# Landmark indices for Right Wrist, Right Shoulder, Right Elbow, Right Eye, and Nose
right_wrist = 16
right_shoulder = 12
right_elbow = 14
right_eye = 5
nose = 0

# Parse the landmarks string into a list of landmarks
def parse_landmarks(input_string):
    pattern = re.compile(r"x: ([\d\-.]+)\ny: ([\d\-.]+)\nz: ([\d\-.]+)\nvisibility: ([\d\-.]+)")
    matches = pattern.findall(input_string)
    landmarks = []
    for match in matches:
        x, y, z, visibility = map(float, match)
        landmarks.append({"x": x, "y": y, "z": z, "visibility": visibility})
    return landmarks

# Function to calculate angle difference between two vectors
def calculate_angle_similarity(vector_a, vector_b):
    if np.linalg.norm(vector_a) == 0 or np.linalg.norm(vector_b) == 0:
        return np.nan  # Handle zero-length vectors
    cosine_similarity = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    angle = np.arccos(np.clip(cosine_similarity, -1.0, 1.0))  # Calculate the angle (in radians)
    return np.degrees(angle)

# Function to calculate Euclidean distance between two points
def calculate_distance(point_a, point_b):
    a = np.array([point_a['x'], point_a['y'], point_a['z']])
    b = np.array([point_b['x'], point_b['y'], point_b['z']])
    return np.linalg.norm(a - b)

# Extract vectors and points for comparison
def extract_landmarks_and_vectors(df):
    vectors = {
        "shoulder_to_wrist_3d": [],
        "elbow_to_wrist_3d": [],
        "eye_to_wrist_3d": [],
        "nose_to_wrist_3d": [],
        'distance': []
    }
    points = {
        "wrist_3d": [],
        "shoulder_3d": [],
        "elbow_3d": [],
        "eye_3d": [],
        "nose_3d": [],
        'distance': []
    }

    for i, row in df.iterrows():
        landmarks_3d = parse_landmarks(row['landmarks_3d'])
        right_wrist_3d = landmarks_3d[right_wrist]
        right_elbow_3d = landmarks_3d[right_elbow]
        right_shoulder_3d = landmarks_3d[right_shoulder]
        right_eye_3d = landmarks_3d[right_eye]
        nose_3d = landmarks_3d[nose]
        
        # Calculate vectors for 3D landmarks
        vectors['shoulder_to_wrist_3d'].append(np.array([right_wrist_3d['x'] - right_shoulder_3d['x'],
                                                         right_wrist_3d['y'] - right_shoulder_3d['y'],
                                                         right_wrist_3d['z'] - right_shoulder_3d['z']]))
        
        vectors['elbow_to_wrist_3d'].append(np.array([right_wrist_3d['x'] - right_elbow_3d['x'],
                                                      right_wrist_3d['y'] - right_elbow_3d['y'],
                                                      right_wrist_3d['z'] - right_elbow_3d['z']]))
        
        vectors['eye_to_wrist_3d'].append(np.array([right_wrist_3d['x'] - right_eye_3d['x'],
                                                    right_wrist_3d['y'] - right_eye_3d['y'],
                                                    right_wrist_3d['z'] - right_eye_3d['z']]))
        
        vectors['nose_to_wrist_3d'].append(np.array([right_wrist_3d['x'] - nose_3d['x'],
                                                     right_wrist_3d['y'] - nose_3d['y'],
                                                     right_wrist_3d['z'] - nose_3d['z']]))
        
        # Append the raw 3D points
        points['wrist_3d'].append(right_wrist_3d)
        points['shoulder_3d'].append(right_shoulder_3d)
        points['elbow_3d'].append(right_elbow_3d)
        points['eye_3d'].append(right_eye_3d)
        points['nose_3d'].append(nose_3d)

        # Store the distance from the target
        vectors['distance'].append(row['m_away'])
        points['distance'].append(row['m_away'])

    return vectors, points

# Compute angle differences, Euclidean distances, and significance tests
def compute_differences_by_distance(vectors, points):
    unique_distances = np.unique(vectors['distance'])
    
    angle_difference_by_distance = {
        "distance": [],
        "shoulder_to_wrist_angle": [],
        "elbow_to_wrist_angle": [],
        "eye_to_wrist_angle": [],
        "nose_to_wrist_angle": []
    }
    
    landmark_difference_by_distance = {
        "distance": [],
        "wrist_distance": [],
        "shoulder_distance": [],
        "elbow_distance": [],
        "eye_distance": [],
        "nose_distance": []
    }

    angle_data_groups = {"shoulder": [], "elbow": [], "eye": [], "nose": []}
    landmark_data_groups = {"wrist": [], "shoulder": [], "elbow": [], "eye": [], "nose": []}

    for dist in unique_distances:
        indices = [i for i, d in enumerate(vectors['distance']) if d == dist]

        # Use the first vector/point as the reference for this group
        reference_vectors = {
            'shoulder_to_wrist': vectors['shoulder_to_wrist_3d'][indices[0]],
            'elbow_to_wrist': vectors['elbow_to_wrist_3d'][indices[0]],
            'eye_to_wrist': vectors['eye_to_wrist_3d'][indices[0]],
            'nose_to_wrist': vectors['nose_to_wrist_3d'][indices[0]]
        }
        
        reference_points = {
            'wrist': points['wrist_3d'][indices[0]],
            'shoulder': points['shoulder_3d'][indices[0]],
            'elbow': points['elbow_3d'][indices[0]],
            'eye': points['eye_3d'][indices[0]],
            'nose': points['nose_3d'][indices[0]]
        }

        # Calculate the angle differences
        shoulder_angle_diffs = [calculate_angle_similarity(reference_vectors['shoulder_to_wrist'], vectors['shoulder_to_wrist_3d'][i]) for i in indices]
        elbow_angle_diffs = [calculate_angle_similarity(reference_vectors['elbow_to_wrist'], vectors['elbow_to_wrist_3d'][i]) for i in indices]
        eye_angle_diffs = [calculate_angle_similarity(reference_vectors['eye_to_wrist'], vectors['eye_to_wrist_3d'][i]) for i in indices]
        nose_angle_diffs = [calculate_angle_similarity(reference_vectors['nose_to_wrist'], vectors['nose_to_wrist_3d'][i]) for i in indices]

        # Calculate the landmark differences (Euclidean distance)
        wrist_diffs = [calculate_distance(reference_points['wrist'], points['wrist_3d'][i]) for i in indices]
        shoulder_diffs = [calculate_distance(reference_points['shoulder'], points['shoulder_3d'][i]) for i in indices]
        elbow_diffs = [calculate_distance(reference_points['elbow'], points['elbow_3d'][i]) for i in indices]
        eye_diffs = [calculate_distance(reference_points['eye'], points['eye_3d'][i]) for i in indices]
        nose_diffs = [calculate_distance(reference_points['nose'], points['nose_3d'][i]) for i in indices]

        # Store the differences
        angle_difference_by_distance['distance'].append(dist)
        angle_difference_by_distance['shoulder_to_wrist_angle'].append(np.nanmean(shoulder_angle_diffs))
        angle_difference_by_distance['elbow_to_wrist_angle'].append(np.nanmean(elbow_angle_diffs))
        angle_difference_by_distance['eye_to_wrist_angle'].append(np.nanmean(eye_angle_diffs))
        angle_difference_by_distance['nose_to_wrist_angle'].append(np.nanmean(nose_angle_diffs))

        landmark_difference_by_distance['distance'].append(dist)
        landmark_difference_by_distance['wrist_distance'].append(np.nanmean(wrist_diffs))
        landmark_difference_by_distance['shoulder_distance'].append(np.nanmean(shoulder_diffs))
        landmark_difference_by_distance['elbow_distance'].append(np.nanmean(elbow_diffs))
        landmark_difference_by_distance['eye_distance'].append(np.nanmean(eye_diffs))
        landmark_difference_by_distance['nose_distance'].append(np.nanmean(nose_diffs))

        # Append to data groups for significance testing
        angle_data_groups["shoulder"].append(shoulder_angle_diffs)
        angle_data_groups["elbow"].append(elbow_angle_diffs)
        angle_data_groups["eye"].append(eye_angle_diffs)
        angle_data_groups["nose"].append(nose_angle_diffs)

        landmark_data_groups["wrist"].append(wrist_diffs)
        landmark_data_groups["shoulder"].append(shoulder_diffs)
        landmark_data_groups["elbow"].append(elbow_diffs)
        landmark_data_groups["eye"].append(eye_diffs)
        landmark_data_groups["nose"].append(nose_diffs)

    # Perform ANOVA to check significance
    significance_results = {
        "angles": {},
        "landmarks": {}
    }

    for key in angle_data_groups:
        F_statistic, p_value = f_oneway(*angle_data_groups[key])
        significance_results['angles'][key] = {"F-statistic": F_statistic, "p-value": p_value}

    for key in landmark_data_groups:
        F_statistic, p_value = f_oneway(*landmark_data_groups[key])
        significance_results['landmarks'][key] = {"F-statistic": F_statistic, "p-value": p_value}

    return angle_difference_by_distance, landmark_difference_by_distance, significance_results

# Compute angle differences and Euclidean distances across all distances
def compute_differences(vectors, points):
    reference_vectors = {
        'shoulder_to_wrist': vectors['shoulder_to_wrist_3d'][0],
        'elbow_to_wrist': vectors['elbow_to_wrist_3d'][0],
        'eye_to_wrist': vectors['eye_to_wrist_3d'][0],
        'nose_to_wrist': vectors['nose_to_wrist_3d'][0]
    }
    
    reference_points = {
        'wrist': points['wrist_3d'][0],
        'shoulder': points['shoulder_3d'][0],
        'elbow': points['elbow_3d'][0],
        'eye': points['eye_3d'][0],
        'nose': points['nose_3d'][0]
    }

    angle_differences = {
        "shoulder_to_wrist_angle": [],
        "elbow_to_wrist_angle": [],
        "eye_to_wrist_angle": [],
        "nose_to_wrist_angle": []
    }
    
    landmark_differences = {
        "wrist_distance": [],
        "shoulder_distance": [],
        "elbow_distance": [],
        "eye_distance": [],
        "nose_distance": []
    }

    angle_data_groups = {"shoulder": [], "elbow": [], "eye": [], "nose": []}
    landmark_data_groups = {"wrist": [], "shoulder": [], "elbow": [], "eye": [], "nose": []}

    for i in range(len(vectors['distance'])):
        # Calculate the angle differences
        shoulder_angle_diff = calculate_angle_similarity(reference_vectors['shoulder_to_wrist'], vectors['shoulder_to_wrist_3d'][i])
        elbow_angle_diff = calculate_angle_similarity(reference_vectors['elbow_to_wrist'], vectors['elbow_to_wrist_3d'][i])
        eye_angle_diff = calculate_angle_similarity(reference_vectors['eye_to_wrist'], vectors['eye_to_wrist_3d'][i])
        nose_angle_diff = calculate_angle_similarity(reference_vectors['nose_to_wrist'], vectors['nose_to_wrist_3d'][i])

        # Calculate the landmark differences (Euclidean distance)
        wrist_diff = calculate_distance(reference_points['wrist'], points['wrist_3d'][i])
        shoulder_diff = calculate_distance(reference_points['shoulder'], points['shoulder_3d'][i])
        elbow_diff = calculate_distance(reference_points['elbow'], points['elbow_3d'][i])
        eye_diff = calculate_distance(reference_points['eye'], points['eye_3d'][i])
        nose_diff = calculate_distance(reference_points['nose'], points['nose_3d'][i])

        # Append to the result lists
        angle_differences['shoulder_to_wrist_angle'].append(shoulder_angle_diff)
        angle_differences['elbow_to_wrist_angle'].append(elbow_angle_diff)
        angle_differences['eye_to_wrist_angle'].append(eye_angle_diff)
        angle_differences['nose_to_wrist_angle'].append(nose_angle_diff)

        landmark_differences['wrist_distance'].append(wrist_diff)
        landmark_differences['shoulder_distance'].append(shoulder_diff)
        landmark_differences['elbow_distance'].append(elbow_diff)
        landmark_differences['eye_distance'].append(eye_diff)
        landmark_differences['nose_distance'].append(nose_diff)

        # Append to data groups for significance testing
        angle_data_groups["shoulder"].append(shoulder_angle_diff)
        angle_data_groups["elbow"].append(elbow_angle_diff)
        angle_data_groups["eye"].append(eye_angle_diff)
        angle_data_groups["nose"].append(nose_angle_diff)

        landmark_data_groups["wrist"].append(wrist_diff)
        landmark_data_groups["shoulder"].append(shoulder_diff)
        landmark_data_groups["elbow"].append(elbow_diff)
        landmark_data_groups["eye"].append(eye_diff)
        landmark_data_groups["nose"].append(nose_diff)

    # Perform ANOVA to check significance
    significance_results = {"angles": {}, "landmarks": {}}
    
    for key in angle_data_groups:
        F_statistic, p_value = f_oneway(*angle_data_groups[key])
        significance_results['angles'][key] = {"F-statistic": F_statistic, "p-value": p_value}
    
    for key in landmark_data_groups:
        F_statistic, p_value = f_oneway(*landmark_data_groups[key])
        significance_results['landmarks'][key] = {"F-statistic": F_statistic, "p-value": p_value}

    return angle_differences, landmark_differences, significance_results

# Main function
def main():
    vectors, points = extract_landmarks_and_vectors(df)

    # Compute angle and distance differences by unique distances
    angle_diffs, landmark_diffs, significance_results = compute_differences_by_distance(vectors, points)

    overall_angle_diffs, _, _ = compute_differences(vectors, points)
    variances = {key: np.var(values, ddof=1) for key, values in overall_angle_diffs.items()}
    mean = {key: np.mean(values) for key, values in overall_angle_diffs.items()}
    
    print('')
    print(overall_angle_diffs)
    # Save the results to CSV
    angle_df = pd.DataFrame(angle_diffs)
    landmark_df = pd.DataFrame(landmark_diffs)
    print(angle_df)
    print(landmark_df)
    angle_df.to_csv('angle_differences.csv', index=False)
    landmark_df.to_csv('landmark_differences.csv', index=False)

    # Print significance results
    print("Angle Significance Results:", significance_results['angles'])
    print("Landmark Significance Results:", significance_results['landmarks'])

    print("Results saved to CSV.")

if __name__ == "__main__":
    main()