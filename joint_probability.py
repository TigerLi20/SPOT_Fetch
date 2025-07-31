import json
import time
import os
from pose_detection.gesture_util import read_json_locked, write_json_locked

def combine_prob():
    """Combines language and gesture probabilities using Laplace smoothing and normalization."""
    detection_json_path = ".tmp/detection_confidence.json"
    gesture_detection_json_path = ".tmp/gesture_confidence.json"
    gesture_json_path = ".tmp/gesture_vector.json"

    # Read JSON files with locking
    data = read_json_locked(detection_json_path)
    gesture_data = read_json_locked(gesture_json_path)
    
    if not data or not gesture_data:
        print("⚠️ Warning: One or both JSON files are empty or missing!")
        return

    cone_confidence = gesture_data.get('pointing_vector_conf', 1.0)  # Default to 1 if missing

    # Laplace smoothing parameter
    a = 1

    # Count valid language targets
    num_lang_targets = sum(1 for mark in data if mark.get('lang_prob', 0) > 0)
    num_gest_targets = sum(1 for mark in data if mark.get('gest_prob', 0) > 0)

    print(f"🟢 Number of language targets: {num_lang_targets}, gesture targets: {num_gest_targets}")

    # Normalize language probability
    if num_lang_targets > 0:
        unnormalized_lang_sum = 0
        for mark in data:
            lang_prob = mark['lang_prob']
            predicted_iou = mark['predicted_iou']
            mark['lang_prob_normalized'] = ((lang_prob + a) / (num_lang_targets + a)) * predicted_iou
            unnormalized_lang_sum += mark['lang_prob_normalized']

        # Final normalization
        for mark in data:
            mark['lang_prob_normalized'] /= unnormalized_lang_sum if unnormalized_lang_sum > 0 else 1

    # Normalize gesture probability
    if num_gest_targets > 0:
        data = read_json_locked(gesture_detection_json_path)
        unnormalized_gest_sum = 0
        for mark in data:
            gest_prob = mark.get('gest_prob', 0)
            mark['gest_prob_normalized'] = ((gest_prob + a) / (num_gest_targets + a)) * cone_confidence
            unnormalized_gest_sum += mark['gest_prob_normalized']

        # Final normalization
        for mark in data:
            mark['gest_prob_normalized'] /= unnormalized_gest_sum if unnormalized_gest_sum > 0 else 1

    # Edge case: No language or gesture detected
    if num_lang_targets == 0 and num_gest_targets == 0:
        print("⚠️ No language or gesture detected. Assigning uniform probability.")
        uniform_prob = 1 / len(data) if data else 0
        for mark in data:
            mark['combined_prob'] = uniform_prob

    # Edge case: Only gesture detected
    elif num_lang_targets == 0 and num_gest_targets > 0:
        print("⚠️ Only gesture detected. Using gesture probability directly.")
        for mark in data:
            mark['combined_prob'] = mark.get('gest_prob_normalized', 0)

    # Edge case: Only language detected
    elif num_lang_targets > 0 and num_gest_targets == 0:
        print("⚠️ Only language detected. Using language probability directly.")
        for mark in data:
            mark['combined_prob'] = mark.get('lang_prob_normalized', 0)

    # Case: Both detected, apply multiplication and normalize
    else:
        print("✅ Both language and gesture detected. Combining probabilities.")
        unnormalized_combined_sum = 0
        for mark in data:
            mark['combined_prob'] = mark.get('lang_prob_normalized', 1) * mark.get('gest_prob_normalized', 1)
            unnormalized_combined_sum += mark['combined_prob']

        # Normalize combined probability
        for mark in data:
            mark['combined_prob'] /= unnormalized_combined_sum if unnormalized_combined_sum > 0 else 1

    # Print results for debugging
    print("\n🔹 Final Combined Probabilities:")
    for mark in data:
        print(f"  - Mark {mark['mark']}: {mark['combined_prob']:.4f}")

    # Write back updated data safely
    write_json_locked(data, detection_json_path, max_retries= 50,
    retry_delay= 0.3)

def main():
    """Continuously updates probabilities every 2 seconds."""
    time.sleep(5)
    while True:
        combine_prob()
        time.sleep(2)

if __name__ == "__main__":
    main()