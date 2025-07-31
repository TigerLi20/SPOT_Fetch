import json
import time
import os
from pose_detection.gesture_util import read_json_locked, write_json_locked

def bayesian_update():
    """Performs a Bayesian update on language and gesture probabilities, while keeping all object attributes intact."""
    detection_json_path = ".tmp/detection_confidence.json"  # Contains detection likelihoods
    gesture_json_path = ".tmp/gesture_confidence.json"  # Contains detection + gesture likelihoods
    conf_json_path = ".tmp/confidence.json"  # Stores final posterior

    # Read JSON files with locking
    detection_data = read_json_locked(detection_json_path)
    gesture_data = read_json_locked(gesture_json_path)
    previous_posterior = read_json_locked(conf_json_path)

    # Check if required data is available
    if not detection_data or not gesture_data:
        print("⚠️ Warning: One or both JSON files are empty or missing!")
        return

    # Initialize prior using previous posterior, or use uniform prior if first frame
    num_objects = len(detection_data)
    prior = {}

    if previous_posterior:
        print("🔄 Using previous posterior as prior.")
        for mark in detection_data:
            mark_id = mark['mark']
            prior[mark_id] = next(
                (p['combined_prob'] for p in previous_posterior if p['mark'] == mark_id), 
                1 / num_objects
            )  # If object wasn't in the previous frame, use uniform probability
    else:
        print("🔄 No previous posterior found, using uniform prior.")
        prior = {mark['mark']: 1 / num_objects for mark in detection_data} if num_objects > 0 else {}

    # Compute likelihoods from language and gesture
    likelihood_lang = {mark['mark']: mark.get('lang_prob', 0.01) for mark in gesture_data}
    likelihood_gest = {mark['mark']: mark.get('gest_prob', 0.01) for mark in gesture_data}

    # Compute the unnormalized posterior
    posterior = {}
    for mark_id in prior:
        posterior[mark_id] = prior[mark_id] * likelihood_lang.get(mark_id, 1) * likelihood_gest.get(mark_id, 1)

    # Normalize posterior
    normalization_factor = sum(posterior.values()) if sum(posterior.values()) > 0 else 1
    for mark_id in posterior:
        posterior[mark_id] /= normalization_factor

    # Construct updated posterior with all attributes
    updated_posterior = []
    for mark in detection_data:
        mark_id = mark['mark']
        updated_mark = mark.copy()  # Start with detection data
        gesture_info = next((g for g in gesture_data if g['mark'] == mark_id), {})  # Find matching gesture info

        updated_mark.update(gesture_info)  # Merge gesture data
        updated_mark['combined_prob'] = posterior[mark_id]  # Update Bayesian probability

        updated_posterior.append(updated_mark)  # Store in list

    # Save updated posterior to confidence.json
    print("\n🔹 Final Bayesian Combined Probabilities:")
    for mark in updated_posterior:
        print(f"  - Mark {mark['mark']}: {mark['combined_prob']:.4f}")

    write_json_locked(updated_posterior, conf_json_path, max_retries=50, retry_delay=0.3)

def main():
    """Continuously updates probabilities every 2 seconds."""
    conf_json_path = ".tmp/confidence.json"
    
    # Delete previous confidence file to start fresh
    if os.path.exists(conf_json_path):
        os.remove(conf_json_path)
        print("🗑️ Deleted previous confidence.json file.")

    time.sleep(5)
    while True:
        bayesian_update()
        time.sleep(2)

if __name__ == "__main__":
    main()