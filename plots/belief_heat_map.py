import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
import time

# Set ROOT_DIR and add to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from pose_detection.gesture_util import read_json_locked

# Path to confidence.json
conf_path = "./.tmp/confidence.json"

# Set fixed X and Y axis ranges (adjust as needed)
X_RANGE = (-10, 0)  # X-axis in meters
Y_RANGE = (-1, 7)  # Y-axis in meters

def safe_read_json(file_path):
    """Reads JSON safely, returning an empty list if file is missing or invalid."""
    try:
        if os.path.exists(file_path):
            return read_json_locked(file_path)
        else:
            print(f"⚠️ Warning: {file_path} not found. Using empty data.")
            return []
    except json.JSONDecodeError as e:
        print(f"❌ Error reading JSON: {e}")
        return []

def plot_heatmap():
    """Plots a heatmap of belief distribution using X-Y object centers and probability, with live updates."""
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Store colorbar reference
    first_run = True
    while True:
        data = safe_read_json(conf_path)

        if not data:
            print("⚠️ No belief data available to plot.")
            time.sleep(2)
            continue  # Retry after a short delay

        # Extract X, Y coordinates, belief values, and annotations
        x_coords = []
        y_coords = []
        belief_values = []
        annotations = []

        for obj in data:
            if "center_3d" in obj and "combined_prob" in obj:
                x, y, _ = obj["center_3d"]  # Get X-Y coordinates from center_3d
                prob = obj["combined_prob"]  # Get belief probability
                
                # Ignore non-target areas (set probability to zero)
                if prob <= 0.01:
                    prob = 0

                x_coords.append(x)
                y_coords.append(y)
                belief_values.append(prob)
                annotations.append(f"{obj['mark']} ({obj['category']})")  # Label format

        if not x_coords or not y_coords:
            print("⚠️ No valid object locations found.")
            time.sleep(2)
            continue  # Retry after a short delay

        # Convert lists to numpy arrays
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        belief_values = np.array(belief_values)

        # Normalize belief values (avoiding division by zero)
        max_belief = np.max(belief_values)
        if max_belief > 0:
            belief_values /= max_belief

        ax.clear()  # Clear the previous plot

        # Create heatmap scatter plot
        scatter = ax.scatter(x_coords, y_coords, c=belief_values, cmap="hot", s=200, alpha=0.75, edgecolors="k")

        # Remove old color bar before adding a new one
        # global cbar  # Ensure global reference
        # Set black background
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        if first_run:
            cbar = plt.colorbar(scatter, ax=ax, label="Belief Probability")  # Create new color bar
            first_run = False

        # Annotate points with mark category
        for i, label in enumerate(annotations):
            ax.text(x_coords[i], y_coords[i], label, fontsize=9, ha="right", va="bottom", color="white", weight="bold")

        # Set fixed axis limits
        ax.set_xlim(X_RANGE)
        ax.set_ylim(Y_RANGE)

        # Labels and title
        ax.set_xlabel("X Coordinate (meters)")
        ax.set_ylabel("Y Coordinate (meters)")
        ax.set_title("🔴 Object Belief Heatmap (Live Confidence Update)")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Update plot
        plt.draw()
        plt.pause(2)  # Refresh every 2 seconds

    plt.ioff()  # Disable interactive mode
    plt.show()

# Run live heatmap visualization
plot_heatmap()