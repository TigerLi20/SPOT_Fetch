import matplotlib.pyplot as plt
import time
import json
import os
import sys
import numpy as np

# Set ROOT_DIR and add to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from pose_detection.gesture_util import read_json_locked

# Paths to JSON files
conf_path = "./.tmp/confidence.json"  # Stores final Bayesian belief update

# Function to safely read JSON data
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

# Initialize last modified time
last_modified_time = os.path.getmtime(conf_path) if os.path.exists(conf_path) else None

# Function to check for file updates and reload JSON
def update_data(file_path, last_modified_time):
    """Checks if the JSON file has been modified and updates data."""
    if os.path.exists(file_path):
        modified_time = os.path.getmtime(file_path)
        if last_modified_time is None or modified_time > last_modified_time:
            print("🔄 JSON file updated, reloading...")
            new_data = safe_read_json(file_path)
            return new_data, modified_time
    return None, last_modified_time  # No update found

# Function to plot live belief updates (Top 10)
def plot_live_updates(keys_to_plot=["gest_prob", "lang_prob", "joint_prob"]):
    """
    Live plots belief updates for objects using `confidence.json`.

    Args:
        keys_to_plot (list): Probability components to visualize.
    """
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))

    global last_modified_time
    data = safe_read_json(conf_path)

    while True:
        new_data, last_modified_time = update_data(conf_path, last_modified_time)
        
        if new_data:
            data = new_data  # Update data only if new data is available
            print("✅ Data updated in plot.")

            # Sort data by `combined_prob` (posterior belief) in descending order, take top 10
            # data = sorted(data, key=lambda x: x.get("joint_prob", 0), reverse=True)[:10]

            ax.clear()  # Clear previous plot
            x = np.arange(len(data))  # X-axis positions

            width = 0.2  # Bar width for grouping
            colors = ['blue', 'green', 'purple']
            offsets = [-width, 0, width]  # Position bars side by side

            for idx, key in enumerate(keys_to_plot):
                y = [item.get(key, 0) for item in data]  # Default to 0 if key is missing
                ax.bar(x + offsets[idx], y, width, label=key, color=colors[idx])  # Plot bars

            ax.set_xticks(x)
            ax.set_xticklabels([str(str(item.get("mark", "Unknown"))+str(item.get("category", "Unknown"))) for item in data], rotation=25)  # Set object labels
            ax.set_title("Live Object Belief Distribution")
            ax.set_xlabel("Object Category")
            ax.set_ylabel("Probability Value")
            ax.legend(["Gesture Likelihood", "Language Likelihood", "Joint prob"], loc="upper right")
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.draw()  # Explicitly redraw the plot
            plt.pause(0.5)  # Pause for updates

    plt.ioff()  # Disable interactive mode
    plt.show()

# Start live plotting
if os.path.exists(conf_path):
    plot_live_updates()
else:
    print("⚠️ No `confidence.json` found. Please run Bayesian update first.")