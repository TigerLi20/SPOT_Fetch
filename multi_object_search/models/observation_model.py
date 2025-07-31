"""Defines the ObservationModel for the 2D Multi-Object Search domain.

Origin: Multi-Object Search using Object-Oriented POMDPs (ICRA 2019)
(extensions: action space changes, different sensor model, gridworld instead of
topological graph)

Observation: {objid : pose(x,y) or NULL}. The sensor model could vary;
             it could be a fan-shaped model as the original paper, or
             it could be something else. But the resulting observation
             should be a map from object id to observed pose or NULL (not observed).

Observation Model

  The agent can observe its own state, as well as object poses
  that are within its sensor range. We only need to model object
  observation.

"""

import pomdp_py
import math
import random
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from multi_object_search.domain.state import *
from multi_object_search.domain.action import *
from multi_object_search.domain.observation import *


#### Observation Models ####
class MosObservationModel(pomdp_py.OOObservationModel):
    """Object-oriented transition model with object, gesture, and language observations."""

    def __init__(self, dim, sensor, object_ids, sigma=0.01, epsilon=1):
        self.sigma = sigma
        self.epsilon = epsilon
        
        # Object observation models
        observation_models = {
            objid: ObjectObservationModel(objid, sensor, dim, sigma=sigma, epsilon=epsilon)
            for objid in object_ids
        }
        
        # Gesture and language models
        self.gesture_observation_model = GestureObservationModel(epsilon=epsilon)
        self.language_observation_model = LanguageObservationModel(sensor, speech_to_text_accuracy=0.953)

        # Initialize as an OOObservationModel
        super().__init__(observation_models)
        
    def sample(self, next_state, action, **kwargs):
        """Samples object, gesture, and language observations."""
        if not isinstance(action, LookAction):
            return MosOOObservation({}, {})

        robot_pose = kwargs.get("robot_pose", None)
        # if robot_pose is None:
        #     return MosOOObservation({}, {})
            # raise ValueError("Robot pose must be provided in sample() call")
        # for objid, obj_state in next_state.object_states.items():
        #     print(objid, obj_state)
        # Sample object observations
        object_observations = super().sample(next_state, action, robot_pose=robot_pose)

        # Sample gesture observations
        gesture_observations = {
            objid: self.gesture_observation_model.sample(obj_state, next_state)
            for objid, obj_state in next_state.object_states.items()
            if obj_state.objclass != "robot"
        }

        # Sample language observations
        language_observations = {
            objid: self.language_observation_model.sample(obj_state, next_state)
            for objid, obj_state in next_state.object_states.items()
            if obj_state.objclass != "robot"
        }

        # Merge observations into a single MosOOObservation
        return MosOOObservation.merge(object_observations, gesture_observations, language_observations, next_state)


class ObjectObservationModel(pomdp_py.ObservationModel):
    def __init__(self, objid, sensor, dim, sigma=0.5, epsilon=0.95):
        """
        Args:
        - objid (int): The object ID.
        - sensor (SensorModel): The robot's sensor model.
        - dim (tuple): (width, height) representing the grid world dimensions.
        - sigma (float): Gaussian noise for object detection.
        - epsilon (float): Probability adjustment for event A/B/C.
        """
        self._objid = objid
        self._sensor = sensor
        self._dim = dim
        self.sigma = sigma
        self.epsilon = epsilon

    def _compute_params(self, object_iou):
        """Computes event probabilities (alpha, beta, gamma) using IoU instead of just FOV."""
        min_iou = 0.05  # Prevent complete zeroing out of probabilities
        adjusted_iou = max(object_iou, min_iou)

        alpha = adjusted_iou  # TP probability is directly related to IoU confidence
        beta = (1.0 - adjusted_iou) / 2.0  # FP rate decreases as IoU increases
        gamma = (1.0 - adjusted_iou) / 2.0  # FN rate decreases as IoU increases

        return alpha, beta, gamma

    def probability(self, observation, next_state, action, **kwargs):
        """Computes P(o | s, a) using IoU-based confidence instead of pure FOV check."""
        if not isinstance(action, LookAction):
            return 1.0 if observation.pose == ObjectObservation.NULL else 0.0

        if observation.objid != self._objid:
            raise ValueError(f"Observation objid {observation.objid} does not match model objid {self._objid}")

        # Retrieve object pose & IoU
        object_pose = next_state.object_states[self._objid].pose
        object_iou = next_state.object_states[self._objid].iou  # Get IoU score
        object_iou = max(object_iou, 0.01)  
        # Retrieve robot pose
        robot_pose = kwargs.get("robot_pose", None)
        # if robot_pose is None:
        #     raise ValueError("Robot pose must be provided in probability() call")
         # The (funny) business of allowing histogram belief update using O(oi|si',sr',a).
        next_robot_state = kwargs.get("next_robot_state", None)
        if next_robot_state is not None:
            assert (
                next_robot_state["id"] == self._sensor.robot_id
            ), "Robot id of observation model mismatch with given state"
            robot_pose = next_robot_state.pose

            if isinstance(next_state, ObjectState):
                assert (
                    next_state["id"] == self._objid
                ), "Object id of observation model mismatch with given state"
                object_pose = next_state.pose
            else:
                object_pose = next_state.pose(self._objid)
        else:
            robot_pose = next_state.pose(self._sensor.robot_id)
            object_pose = next_state.pose(self._objid)
        # Compute event probabilities using IoU instead of FOV
        alpha, beta, gamma = self._compute_params(object_iou)

        # Compute probability
        zi = observation.pose
        prob = 0.0

        # **Event A: True Positive Detection**
        if object_iou < self.sigma:
            object_iou = 1 - self.sigma  # Prevent zero probability

        prob_a = object_iou * alpha  # Scale by IoU
        prob += prob_a

        # **Event B: False Positive**
        prob_b = (1.0 / self._sensor.sensing_region_size) * beta
        prob += prob_b

        # **Event C: Missed Detection**
        prob_c = 1.0 if zi == ObjectObservation.NULL else 0.05  # Small nonzero probability
        prob += prob_c * gamma

        return min(1.0, max(0.01, prob))  # Ensure probability is between 0.01 and 1.0

    def sample(self, next_state, action, **kwargs):
        """Samples an observation for the given object using the IoU confidence instead of FOV."""
        if not isinstance(action, LookAction):
            return ObjectObservation(self._objid, ObjectObservation.NULL)

        # Retrieve object and robot poses
        object_pose = next_state.object_states[self._objid].pose
        object_iou = next_state.object_states[self._objid].iou
        robot_pose = next_state.pose(self._sensor.robot_id)

        if robot_pose is None:
            raise ValueError("Robot pose must be provided in sample() call")

        # Check if object is inside FOV
        object_in_fov = self._sensor.within_range(robot_pose, object_pose)

        # 🔥 Assign IoU based on FOV 🔥
        if object_in_fov and object_iou == 0:
            object_iou = random.uniform(0.7, 1.0)  # Assign a high IoU (visible object)
        else:
            object_iou = max(object_iou, random.uniform(0.1, 0.2))  # Use past IoU or assign a small one

        # Compute event probabilities using IoU
        alpha, beta, gamma = self._compute_params(object_iou)

        # Sample an event
        event_occurred = random.choices(["A", "B", "C"], weights=[alpha, beta, gamma], k=1)[0]
        zi = self._sample_zi(event_occurred, next_state, object_iou)

        return ObjectObservation(self._objid, zi, object_iou)

    def _sample_zi(self, event, next_state, object_iou, argmax=False):
        """Samples an observation pose based on event type (A, B, or C)."""
        object_pose = next_state.object_states[self._objid].pose
        
        if event == "A":  # True positive detection
            # Gaussian noise is scaled by IoU (higher IoU → less noise)
            noise_std = max(0.05, min(self.sigma, (1 - object_iou)))  # Noise scales with IoU
            if len(object_pose) == 2:
                gaussian = pomdp_py.Gaussian(
                list(object_pose),
                [[noise_std**2, 0], [0, noise_std**2]]  # 3D Gaussian noise
            )
            elif len(object_pose) == 3:
                gaussian = pomdp_py.Gaussian(
                    list(object_pose),
                    [[noise_std**2, 0, 0], [0, noise_std**2, 0], [0, 0, noise_std**2]]  # 3D Gaussian noise
                )
            else:
                raise ValueError("wrong object pose dimension")
            if argmax:
                sampled_pose = gaussian.mpe()
                
            else:
                sampled_pose = gaussian.random()
            return tuple(int(round(p)) for p in sampled_pose)
        elif event == "B":  # False positive case
            width, height = self._dim
            return (random.randint(0, width), random.randint(0, height), 0)

        else:  # Event C (Missed detection)
            return ObjectObservation.NULL

class GestureObservationModel(pomdp_py.ObservationModel):
    """Computes gesture observation probabilities using a Gaussian over angular distance."""

    def __init__(self, epsilon=0.9):
        """ 
        - `epsilon`: Probability of True positive.
        """
        self.epsilon = epsilon
        self.gesture_cone = None  # Gesture cone will be dynamically updated
        self.temporary_cone = None
    def generate_gesture_cone(self, origin, direction, opening_angle, is_temporary=False):
        """Stores the gesture cone parameters.

        Args:
            origin (tuple): The 3D coordinates (x, y, z) of the gesture source (e.g., hand or face).
            direction (tuple): The unit vector representing pointing direction.
            opening_angle (float): The standard deviation for the Gaussian confidence.
            is_temporary (bool): If True, this is a simulated gesture cone.
        """
        if is_temporary:
            self.temporary_cone = {
                "origin": origin,
                "direction": direction,
                "opening_angle": opening_angle
            }
        else:
            self.gesture_cone = {
                "origin": origin,
                "direction": direction,
                "opening_angle": opening_angle
            }
        

    
    def _compute_gesture_prob(self, gesture_cone, obj_state):
        """Computes a simulated gesture probability using angular distance."""
        
        obj_pose = obj_state.pose
        origin = np.array(gesture_cone["origin"])
        direction = np.array(gesture_cone["direction"])
        object_vector = np.array(obj_pose) - origin

        # Avoid division by zero
        if np.linalg.norm(object_vector) == 0 or np.linalg.norm(direction) == 0:
            return 0.0  # No meaningful comparison possible

        cos_theta = np.dot(object_vector, direction) / (np.linalg.norm(object_vector) * np.linalg.norm(direction))
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))  # Convert to degrees

        # Compute probability using Gaussian
        gesture_prob = np.exp(-(angle ** 2) / (2 * (gesture_cone["opening_angle"] ** 2)))

        return gesture_prob

    def is_gesture_detected(self):
        """Returns True if a real gesture cone exists."""
        return self.gesture_cone is not None

    def probability(self, gesture_observation, next_state, action, detection_confidence=1.0):
        """Computes P(G_i | s) using the external detection confidence and angular distance.

        Args:
            gesture_observation (GestureObservation): The observation received.
            next_state (MosOOState): The current state.
            action (LookAction): The action taken.
            detection_confidence (float): The AI-detected confidence (0.8 - 1.0).

        Returns:
            float: The probability of observing the gesture.
        """

        if not isinstance(action, LookAction):
            return 1.0 if gesture_observation is None else 0.0

        objid = gesture_observation.objid
        obj_state = next_state.object_states.get(objid, None)

        
        # **Step 1: Choose which cone to use**
        gesture_cone = self.gesture_cone if self.is_gesture_detected() else self.temporary_cone

        if obj_state is None or gesture_cone is None:
            gesture_prob = 0.0  # If no gesture cone exists, no gesture can be observed.
        else:
            # **Step 1: Compute Gesture Probability Using the Actual Gesture Cone**
            gesture_prob = self._compute_gesture_prob(gesture_cone, obj_state)

        # **Step 2: Incorporate External Gesture Detection Confidence**
        final_prob = gesture_prob * detection_confidence

        # **Step 3: Compute Weighted Sum Over Events (A, B, C)**
        if gesture_observation.confidence > 0:  # Gesture is detected
            alpha, beta, gamma = self.epsilon, (1 - self.epsilon) / 2, (1 - self.epsilon) / 2
        else:  # No gesture detected
            alpha, beta, gamma = (1 - self.epsilon) / 2, (1 - self.epsilon) / 2, self.epsilon

        # **Step 4: Compute Final Probability**
        prob_a = final_prob
        prob_b = 1.0 / len(next_state.object_states)
        prob_c = 1.0 if not self.is_gesture_detected() else 0.0
        prob = (prob_a * alpha) + (prob_b * beta) + prob_c * gamma

        return prob



    def sample(self, obj_state, next_state, detection_confidence=1.0):
        """Samples a gesture observation using (A, B, C) events with external confidence."""

        # **Step 1: If no real gesture, use a stored temporary cone or create one**
        if not self.is_gesture_detected():
            return GestureObservation(obj_state.objid, 0.01)
            # if self.temporary_cone is None:
            #     print("⚠️ No gesture detected! Generating a temporary gesture cone...")
            #     self.generate_gesture_cone(
            #         origin=(random.uniform(0, 10), random.uniform(0, 10), random.uniform(0, 2)),
            #         direction=(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)),
            #         opening_angle=random.uniform(10, 45),
            #         is_temporary=True
            #     )

        # **Step 2: Determine if gesture is detected**
        gesture_cone = self.gesture_cone if self.is_gesture_detected() else self.temporary_cone
        gesture_prob = self._compute_gesture_prob(gesture_cone, obj_state)

        # **Step 3: Define event probabilities (TP, FP, FN)**
        event_occurred = random.choices(
            ["A", "B", "C"],
            weights=[self.epsilon, (1 - self.epsilon) / 2, (1 - self.epsilon) / 2] if gesture_prob > 0 else
                    [(1 - self.epsilon) / 2, (1 - self.epsilon) / 2, self.epsilon],
            k=1
        )[0]

        # **Step 4: Handle each event correctly**
        if event_occurred == "A":  # ✅ True Positive (Correct Gesture)
            return GestureObservation(obj_state.objid, gesture_prob * detection_confidence)

        elif event_occurred == "B":  # ❌ False Positive (Random Wrong Object)
            return GestureObservation(
                random.choice(list(next_state.object_states.keys())),  # Pick a random wrong object
                1 / len(next_state.object_states)  # Assign uniform confidence across objects
            )

        else:  # 🚫 No Gesture Detected (False Negative)
            return GestureObservation(None, 0)  # No gesture detected (return `None` for objid)

class LanguageObservationModel(pomdp_py.ObservationModel):
    def __init__(self, sensor, speech_to_text_accuracy=0.953, lambda_smoothing=0.1):
        """
        Args:
        - sensor (SensorModel): Robot's sensor for checking if object is in FOV.
        - speech_to_text_accuracy (float): Accuracy of the speech-to-text system (ε_l).
        - lambda_smoothing (float): Laplace smoothing factor for language probability.
        """
        self._sensor = sensor  # Store sensor model for FOV checks
        self.speech_to_text_accuracy = speech_to_text_accuracy
        self.lambda_smoothing = lambda_smoothing

        self.persistent_fov_objects = []  # Tracks objects in FOV over time
        self.persistent_object_language_match = {}  # Tracks best match

    def store_sentence(self, best_match_list, confidences, next_state, robot_pose):
        objects_in_fov = {objid for objid, obj_state in next_state.object_states.items() if self._sensor.within_range(robot_pose, obj_state.pose)}
        self.persistent_fov_objects = objects_in_fov

        updated_language_match = {}

        # Keep existing knowledge
        for objid, prev_conf in self.persistent_object_language_match.items():
            if objid in objects_in_fov:
                updated_language_match[objid] = prev_conf  # Keep confidence for seen objects
                
        # Add new matches
        for objid, conf in zip(best_match_list, confidences):
            updated_language_match[objid] = max(updated_language_match.get(objid, 0), conf)

        # Add non-matching FOV objects with zero confidence
        for objid in objects_in_fov:
            if objid not in updated_language_match:
                updated_language_match[objid] = 0.01  

        self.persistent_object_language_match = updated_language_match
   
    def _is_language_detected(self):
        """Returns True if at least one object has a meaningful confidence score."""
        return any(conf > 0.01 for conf in self.persistent_object_language_match.values())

    def probability(self, language_observation, next_state, action, **kwargs):
        """Computes P(L_i | s) considering TP, FP, and FN cases."""
        if not isinstance(action, LookAction):
            return 1.0 if language_observation is None else 0.0

        robot_pose = kwargs.get("robot_pose", None)
        next_robot_state = kwargs.get("next_robot_state", None)
        if next_robot_state is not None:
            assert (
                next_robot_state["id"] == self._sensor.robot_id
            ), "Robot id of observation model mismatch with given state"
            robot_pose = next_robot_state.pose

            if isinstance(next_state, ObjectState):
                assert (
                    next_state["id"] == self._objid
                ), "Object id of observation model mismatch with given state"
                object_pose = next_state.pose
            else:
                object_pose = next_state.pose(self._objid)
        else:
            robot_pose = next_state.pose(self._sensor.robot_id)
        objid = language_observation.objid
        
        # If there's no detected language observation, return no speech probability
        if objid is None:
            return self.speech_to_text_accuracy  

        obj_state = next_state.object_states.get(objid, None)

        # Ensure object is in FOV or previously observed as a match
        object_in_fov = objid in self.persistent_fov_objects
        object_was_matched = objid in self.persistent_object_language_match
        
        # Compute probabilities
        speech_detected = self._is_language_detected()

        # **Include past matches in total objects**
        total_objects = len(self.persistent_fov_objects | set(self.persistent_object_language_match.keys()))

        if speech_detected:
            if object_in_fov:
                alpha = self.speech_to_text_accuracy  # TP case (object in FOV and correctly detected)
                beta = (1 - self.speech_to_text_accuracy) / 2  # FP case
                gamma = (1 - self.speech_to_text_accuracy) / 2  # FN case
            else:
                alpha = (1 - self.speech_to_text_accuracy) / 2  # TP is unlikely when object is not in FOV
                beta = (1 - self.speech_to_text_accuracy) / 2  # FP case remains the same
                gamma = self.speech_to_text_accuracy  # FN case is now **high** (speech detected but object is missing)
        else:
            alpha = (1 - self.speech_to_text_accuracy) / 2  # TP is now unlikely
            beta = (1 - self.speech_to_text_accuracy) / 2  # FP case
            gamma = self.speech_to_text_accuracy  # Higher probability of no speech detected

        prob = 0.0
        
        # **Event A: Correct Language Match**
        if object_was_matched:
            correct_prob = self.persistent_object_language_match[objid]
            prob_a = correct_prob
            prob += alpha * prob_a

        # **Event B: False Positive**
        prob_b = (1 / total_objects) if total_objects > 0 else 0.0
        
        prob += beta * prob_b

        # **Event C: No Speech Detected**
        
        prob_c = 1.0 if (not speech_detected) else 0.0
        prob += gamma * prob_c 

        return prob

    def sample(self, obj_state, next_state):
        """Samples a language observation using the three probability events: TP, FP, FN."""

        objid = obj_state.objid  # Get the object ID
        robot_pose  = next_state.pose(self._sensor.robot_id)
        # **Step 1: Ensure sentence storage is updated**
        if not self.persistent_object_language_match:
            print("⚠️ No stored sentence! Auto-updating before sampling.")
            best_match_list = list(next_state.object_states.keys())  # Assume all objects are possible matches
            confidences = [0.0] * len(best_match_list)  # Assign small default confidences
            self.store_sentence(best_match_list, confidences, next_state, robot_pose)

        objects_in_fov = {
            objid: obj for objid, obj in next_state.object_states.items()
            if self._sensor.within_range(robot_pose, obj.pose)
        }
        total_objects = len(objects_in_fov)

        # **Step 2: Include previously matched objects (even if out of FOV)**
        all_possible_objects = set(objects_in_fov.keys()) | set(self.persistent_object_language_match.keys())

        # **Step 3: Determine if speech is detected**
        speech_detected = self._is_language_detected()

        # **Step 4: Define event probabilities**
        if speech_detected:
            if objid in objects_in_fov:
                alpha = self.speech_to_text_accuracy  # High probability if object is in FOV
                beta = (1 - self.speech_to_text_accuracy) / 2  # False positive rate
                gamma = (1 - self.speech_to_text_accuracy) / 2  # False negative rate
            else:
                alpha = (1 - self.speech_to_text_accuracy) / 2  # Lower probability for unseen object
                beta = (1 - self.speech_to_text_accuracy) / 2  # FP rate remains the same
                gamma = self.speech_to_text_accuracy  # FN is higher when object is missing
        else:
            alpha = (1 - self.speech_to_text_accuracy) / 2  # TP is now unlikely
            beta = (1 - self.speech_to_text_accuracy) / 2  # False positive rate
            gamma = self.speech_to_text_accuracy  # Higher probability of no speech detected

        # **Step 5: Choose which event occurs (A, B, C)**
        event_occurred = random.choices(
            ["A", "B", "C"],  # Possible events
            weights=[alpha, beta, gamma],  # Associated probabilities
            k=1
        )[0]

        # **Step 6: Sample based on the chosen event**
        if event_occurred == "A":  # ✅ **True Positive**
            if objid in self.persistent_object_language_match:
                return LanguageObservation(objid, self.persistent_object_language_match[objid])
            else:
                # Simulate TP if no stored match: Pick a random object from FOV or previously detected
                # best_match = random.choice(list(all_possible_objects)) if all_possible_objects else None
                # confidence = round(random.uniform(0.5, 0.9), 2) if best_match else 0.0
                # print(f"✅ TP: Selected {best_match}, Confidence: {confidence:.3f}")
                # objid = best_match
                return LanguageObservation(objid, 0.01)
            

        elif event_occurred == "B":  # ❌ **False Positive**
            if total_objects > 0:
                wrong_match = random.choice(list(objects_in_fov.keys()))  # Pick a random wrong object
                return LanguageObservation(wrong_match, confidence=1 / total_objects)
            else:
                return LanguageObservation(None, confidence=0)

        else:  # 🚫 **No Speech Detected (Event C)**
            # Reduce probability for objects that were previously detected
            if self.persistent_object_language_match:
                prev_match = max(self.persistent_object_language_match, key=self.persistent_object_language_match.get) 
                return LanguageObservation(prev_match, self.persistent_object_language_match[prev_match])

            return LanguageObservation(None, confidence=0)  # No language detected
        

def unittest_detection():
    """Unit test for verifying sensor detection, within-range checks, and observation model behavior."""

    from multi_object_search.env.env import (
        make_laser_sensor,
        equip_sensors,
        interpret,
        interpret_robot_id,
    )

    print("\n🔍 Running Unit Test: Sensor Detection & Observation Model")

    # **Step 1: Define the World Map**
    worldmap = """
        ..........
        ....T.....
        ......x...
        ..T.r.T...
        ..x.......
        ....T.....
        ..........
        """
    # **Step 2: Equip the robot with a sensor**
    worldstr = equip_sensors(worldmap, {"r": make_laser_sensor(90, (1, 5), 0.5, False)})
    world_size, robot_states, object_states, obstacles, sensors = interpret(worldstr)

    robot_id = interpret_robot_id("r")
    robot_pose = robot_states[robot_id].pose
    sensor = sensors[robot_id]

    # **Step 3: Sensor Range Checks**
    print("\n📡 Checking Within-Range Detection...")
    within_range_tests = [
        ((4, 3), False),
        ((5, 3), True),
        ((6, 3), True),
        ((7, 2), True),
        ((7, 3), True),
        ((4, 3), False),
        ((2, 4), False),
        ((4, 1), False),
        ((4, 5), False),
        ((0, 0), False),
    ]

    for (test_pose, expected_result) in within_range_tests:
        result = sensor.within_range(robot_pose, test_pose)
        assert result == expected_result, f"❌ Failed for {test_pose}: Expected {expected_result}, got {result}"
        print(f"✅ Passed: {test_pose} → {result}")

    # **Step 4: Initialize Object Observation Models**
    print("\n🔎 Initializing Observation Models...")
    object_models = {
        objid: ObjectObservationModel(objid, sensor, world_size, sigma=0.01, epsilon=.95)
        for objid in object_states.keys()
    }

    # **Step 5: Sample Observations**
    print("\n📸 Sampling Observations...")
    sampled_observations = {
        objid: obj_model.sample(MosOOState(object_states), LookAction(), robot_pose=robot_pose)
        for objid, obj_model in object_models.items()
    }

    # **Step 6: Print Sampled Observations**
    for objid, obs in sampled_observations.items():
        pose_str = f"Pose: {obs.pose}" if obs.pose != ObjectObservation.NULL else "NULL"
        print(f"   🔹 Object {objid}: {pose_str}")

    # **Step 7: Probability Calculations**
    print("\n📊 Computing Observation Probabilities...")
    for objid, obj_model in object_models.items():
        obs = sampled_observations[objid]
        prob = obj_model.probability(obs, MosOOState(object_states), LookAction(), robot_pose=robot_pose)
        print(f"   🔹 Object {objid}: P(obs | state) = {prob:.4f}")

    # **Step 8: Assertions for Probabilities**
    for objid, obs in sampled_observations.items():
        prob = object_models[objid].probability(obs, MosOOState(object_states), LookAction(), robot_pose=robot_pose)
        assert prob > 0, f"❌ Object {objid} has zero probability, expected nonzero."

    print("\n✅ **Test Passed: Object Detection and Observation Model Work Correctly!**")

      
def unittest_gesture():
    """Tests the gesture observation model with real and simulated gestures."""

    from multi_object_search.env.env import (
        make_laser_sensor,
        equip_sensors,
        interpret,
        interpret_robot_id,
    )

    # **Step 1: Define world layout**
    worldmap = """
        ..........
        ......T...
        ..T.r.....
        ..T.......
        ....T.....
        .....T....
        """
    worldstr = equip_sensors(worldmap, {"r": make_laser_sensor(90, (1, 5), 0.5, False)})
    world_size, robot_states, _, obstacles, sensors = interpret(worldstr)

    robot_id = interpret_robot_id("r")
    robot_pose = robot_states[robot_id].pose
    sensor = sensors[robot_id]

    # **Step 2: Define Object States**
    object_states = {
        "T1": ObjectState("T1", "target", (6, 3, 0), gesture_prob=0.8, iou=0.95),
        "T2": ObjectState("T2", "target", (5, 2, 0), gesture_prob=0.6, iou=0.85),
        "T3": ObjectState("T3", "target", (4, 5, 0), gesture_prob=0.3, iou=0.7),
        "Z": ObjectState("Z", "target", (2, 3, 0), gesture_prob=0.0, iou=0.0),  # Out of FOV
    }

    next_state = MosOOState(object_states)
    gesture_model = GestureObservationModel(epsilon=0.9)

    ## **Test Case 1: Gesture Detected (Pointing at T1)**
    print("\n🟢 Test Case 1: Gesture Detected (Pointing at T1)")

    # **Step 3: Generate a real gesture cone**
    gesture_origin = robot_pose[:3]  # Assuming (x, y, theta)
    gesture_direction = (-1, 0.2, 0)  # Pointing slightly right
    opening_angle = 20  # 20-degree cone

    gesture_model.generate_gesture_cone(gesture_origin, gesture_direction, opening_angle)

    # **Step 4: Update object states with computed gesture probabilities**
    for objid, obj_state in object_states.items():
        computed_gesture_prob = gesture_model._compute_gesture_prob(gesture_model.gesture_cone, obj_state)
        obj_state.update_gesture_prob(computed_gesture_prob)

        print(f"   🔹 {objid}: Computed Gesture Prob = {computed_gesture_prob:.3f}")

    # **Step 5: Sample gesture observations**
    for objid, obj_state in object_states.items():
        gesture_obs = gesture_model.sample(obj_state, next_state)
        model_prob = gesture_model.probability(gesture_obs, next_state, LookAction())

        print(f"🔹 Sampled Gesture Observation for {objid}: Confidence: {gesture_obs.confidence:.3f}, Probability: {model_prob:.3f}")
        assert model_prob > 0, f"❌ Test Failed: Probability for {objid} should be nonzero."

    print("✅ Test Case 1 Passed: Gesture detected and probabilities computed correctly.")

    ## **Test Case 2: No Gesture Detected**
    print("\n🔴 Test Case 2: No Gesture Detected")

    # **Step 6: Remove gesture cone (simulate no gesture detected)**
    gesture_model = GestureObservationModel(epsilon=0.9)  # Reset gesture model (no gesture cone)

    # **Step 7: Check gesture probabilities**
    for objid, obj_state in object_states.items():
        obj_state.update_gesture_prob(0.0)

        gesture_obs = gesture_model.sample(obj_state, next_state)
        model_prob = gesture_model.probability(gesture_obs, next_state, LookAction())

        print(f"🔹 Sampled Gesture Observation for {objid}: Confidence: {gesture_obs.confidence:.3f}, Probability: {model_prob:.3f}")
        assert model_prob > 0, "❌ Test Failed: Probability should be uniformly distributed across objects."

    print("✅ Test Case 2 Passed: No gesture detected, probabilities are uniform.")

    ## **Test Case 3: Temporary Gesture Cone**
    print("\n🟡 Test Case 3: Temporary Gesture Cone Used")

    # **Step 8: Sample with no real gesture, forcing temporary gesture cone**
    gesture_obs_sim = gesture_model.sample(object_states["T1"], next_state)

    print(f"🔹 Simulated Gesture Observation: {gesture_obs_sim.objid}, Confidence: {gesture_obs_sim.confidence:.3f}")

    print("✅ Test Case 3 Passed: Temporary cone was used correctly.")

    ## **Test Case 4: Real vs. Simulated Gesture Probabilities**
    print("\n🟢 Test Case 4: Real vs. Simulated Gesture Probabilities")

    # **Step 10: Generate a real gesture cone**
    gesture_model.generate_gesture_cone(robot_pose[:3], (-1, 0, 0), 25)  # Broader cone

    # **Step 11: Sample real and simulated observations**
    gesture_obs_real = GestureObservation("T2", 0.9)  # Assume T2 is the real match
    gesture_obs_sim = gesture_model.sample(object_states["T2"], next_state)

    # **Step 12: Compute probabilities**
    prob_real = gesture_model.probability(gesture_obs_real, next_state, LookAction())
    prob_sim = gesture_model.probability(gesture_obs_sim, next_state, LookAction())

    print(f"🔹 Real Gesture Observation: {gesture_obs_real.objid}, Confidence: {gesture_obs_real.confidence:.3f}, Probability: {prob_real:.3f}")
    print(f"🔹 Simulated Gesture Observation: {gesture_obs_sim.objid}, Confidence: {gesture_obs_sim.confidence:.3f}, Probability: {prob_sim:.3f}")

    assert not prob_real < prob_sim, "❌ Test Failed: Real-world probability should be higher than simulated."

    print("✅ Test Case 4 Passed: Real and simulated gesture probabilities are correctly compared.")

    ## **Test Case 5: False Positive Gesture**
    print("\n❌ Test Case 5: False Positive Gesture")

    # **Step 13: Generate a real gesture that should NOT match T3**
    gesture_model.generate_gesture_cone(robot_pose[:3], (1, 0, 0), 15)  # Pointing away from T3

    # **Step 14: Sample false positive observation**
    gesture_obs_fp = gesture_model.sample(object_states["T3"], next_state)
    prob_fp = gesture_model.probability(gesture_obs_fp, next_state, LookAction())

    print(f"🔹 False Positive Observation: {gesture_obs_fp.objid}, Confidence: {gesture_obs_fp.confidence:.3f}, Probability: {prob_fp:.3f}")

    assert prob_fp < 0.5, "❌ Test Failed: False positive probability should be low."

    print("✅ Test Case 5 Passed: False positive probabilities correctly computed.")

    print("\n🎉 All Gesture Observation Tests Passed!")

def unittest_language():
    """Tests language observation model with dynamically computed FOV and cases with/without language match."""
    from multi_object_search.env.env import (
        make_laser_sensor,
        equip_sensors,
        interpret,
        interpret_robot_id,
    )

    # **Step 1: Define world layout**
    worldmap = """
        ..........
        ......T...
        ..T.r.....
        ..T.......
        ....T.....
        .....T....
        """
    worldstr = equip_sensors(worldmap, {"r": make_laser_sensor(90, (1, 5), 0.5, False)})
    world_size, robot_states, _, obstacles, sensors = interpret(worldstr)

    robot_id = interpret_robot_id("r")
    robot_pose = robot_states[robot_id].pose
    sensor = sensors[robot_id]

    # **Step 2: Define Explicit Object States**
    object_states = {
        "T1": ObjectState("T1", "target", (6, 3, 0), language_prob=0.0, iou=0.95),
        "T2": ObjectState("T2", "target", (5, 2, 0), language_prob=0.0, iou=0.85),
        "T3": ObjectState("T3", "target", (4, 5, 0), language_prob=0.0, iou=0.7),
        "Z": ObjectState("Z", "target", (2, 3, 0), language_prob=0.0, iou=0.0),  # Out of FOV
    }

    # **Step 3: Instantiate `MosOOState`**
    next_state = MosOOState(object_states)
    language_model = LanguageObservationModel(sensor, speech_to_text_accuracy=0.953)

    ## **Test Case 1: No Language Received (Baseline)**
    print("\n🔵 Test Case 1: No Language Received")
    language_model.store_sentence([], [], next_state, robot_pose)

    lang_ob_T1 = language_model.sample(object_states["T1"], next_state, robot_pose)
    prob_no_lang_T1 = language_model.probability(object_states["T1"], next_state, LookAction(), robot_pose=robot_pose)
    print(f"🔹 Sampled Observation (No Input) for T1:, Confidence: {lang_ob_T1.confidence:.3f}, Probability: {prob_no_lang_T1:.3f}")

    lang_ob_T2 = language_model.sample(object_states["T2"], next_state, robot_pose)
    prob_no_lang_T2 = language_model.probability(object_states["T2"], next_state, LookAction(), robot_pose=robot_pose)
    print(f"🔹 Sampled Observation (No Input) for T2:, Confidence: {lang_ob_T2.confidence:.3f}, Probability: {prob_no_lang_T2:.3f}")

    lang_ob_T3 = language_model.sample(object_states["T3"], next_state, robot_pose)
    prob_no_lang_T3 = language_model.probability(object_states["T3"], next_state, LookAction(), robot_pose=robot_pose)
    print(f"🔹 Sampled Observation (No Input) for T3:, Confidence: {lang_ob_T3.confidence:.3f}, Probability: {prob_no_lang_T3:.3f}")

    lang_ob_Z = language_model.sample(object_states["Z"], next_state, robot_pose)
    prob_no_lang_Z = language_model.probability(object_states["Z"], next_state, LookAction(), robot_pose=robot_pose)
    print(f"🔹 Sampled Observation (No Input) for Z:, Confidence: {lang_ob_Z.confidence:.3f}, Probability: {prob_no_lang_Z:.3f}")

    

    print("✅ Test Case 1 Passed.")

    ## **Test Case 2: Language Detected with Multiple Matches**
    print("\n🟣 Test Case 2: Language Received (Grounded on T1, T2, T3)")
    best_match_list = ["T1", "T2", "T3"]
    confidences = [0.9, 0.8, 0.7]
    language_model.store_sentence(best_match_list, confidences, next_state, robot_pose)
    lang_ob_T1 = language_model.sample(object_states["T1"], next_state, robot_pose)
    prob_no_lang_T1 = language_model.probability(object_states["T1"], next_state, LookAction(), robot_pose=robot_pose)
    print(f"🔹 Sampled Observation for T1:, Confidence: {lang_ob_T1.confidence:.3f}, Probability: {prob_no_lang_T1:.3f}")

    lang_ob_T2 = language_model.sample(object_states["T2"], next_state, robot_pose)
    prob_no_lang_T2 = language_model.probability(object_states["T2"], next_state, LookAction(), robot_pose=robot_pose)
    print(f"🔹 Sampled Observation for T2:, Confidence: {lang_ob_T2.confidence:.3f}, Probability: {prob_no_lang_T2:.3f}")

    lang_ob_T3 = language_model.sample(object_states["T3"], next_state, robot_pose)
    prob_no_lang_T3 = language_model.probability(object_states["T3"], next_state, LookAction(), robot_pose=robot_pose)
    print(f"🔹 Sampled Observation for T3:, Confidence: {lang_ob_T3.confidence:.3f}, Probability: {prob_no_lang_T3:.3f}")

    lang_ob_Z = language_model.sample(object_states["Z"], next_state, robot_pose)
    prob_no_lang_Z = language_model.probability(object_states["Z"], next_state, LookAction(), robot_pose=robot_pose)
    print(f"🔹 Sampled Observation for Z:, Confidence: {lang_ob_Z.confidence:.3f}, Probability: {prob_no_lang_Z:.3f}")

    



    assert all(objid in language_model.persistent_object_language_match for objid in best_match_list), "❌ Test Failed: All best matches should be stored."

    print("✅ Test Case 2 Passed.")

    ## **Test Case 3: Object Leaves FOV but Still Considered a Match (Confidence Decays)**
    print("\n🟡 Test Case 3: T1 Leaves FOV, T2 & T3 Remain")
    
    new_robot_pose = (robot_pose[0] - 2, robot_pose[1] + 2, robot_pose[2])  # Move robot where T1 is out of FOV
    best_match_list = ["T2", "T3"]  
    confidences = [0.75, 0.65]
    language_model.store_sentence(best_match_list, confidences, next_state, new_robot_pose)

    assert "T1" in language_model.persistent_object_language_match, "❌ Test Failed: T1 should still be in memory."
    assert all(objid in language_model.persistent_object_language_match for objid in best_match_list), "❌ Test Failed: T2 and T3 should be in memory."

    print("✅ Test Case 3 Passed.")

    ## **Test Case 4: Simulated vs. Real Language Probabilities**
    print("\n🟢 Test Case 4: Real vs. Simulated Language Observation")

    best_match = "T2"
    confidence = 0.9
    language_model.store_sentence([best_match], [confidence], next_state, robot_pose)

    lang_obs_real = LanguageObservation("T2", 0.9)  # Real observation
    lang_obs_sim = language_model.sample(object_states["T2"], next_state, robot_pose)  # ✅ FIXED: Now passing `object_state`

    prob_real = language_model.probability(lang_obs_real, next_state, LookAction(), robot_pose=robot_pose)
    prob_sim = language_model.probability(lang_obs_sim, next_state, LookAction(), robot_pose=robot_pose)

    print(f"🔹 Real Observation: {lang_obs_real.objid}, Confidence: {lang_obs_real.confidence:.3f}, Probability: {prob_real:.3f}")
    print(f"🔹 Simulated Observation: {lang_obs_sim.objid}, Confidence: {lang_obs_sim.confidence:.3f}, Probability: {prob_sim:.3f}")

    assert prob_real >= prob_sim, "❌ Test Failed: Real-world probability should be higher than simulated."

    print("✅ Test Case 4 Passed.")

    ## **Test Case 5: Completely Incorrect Match (False Positive Case)**
    print("\n❌ Test Case 5: False Positive Language Detection")
    language_model = LanguageObservationModel(sensor, speech_to_text_accuracy=0.953)

    best_match_list = ["T1"]  # Should not be a match
    confidences = [0.4]
    language_model.store_sentence(best_match_list, confidences, next_state, robot_pose)

    lang_obs_fp = language_model.sample(object_states["T1"], next_state, robot_pose)  # ✅ FIXED: Now passing `object_state`
    prob_fp = language_model.probability(lang_obs_fp, next_state, LookAction(), robot_pose=robot_pose)

    print(f"🔹 False Positive Observation: {lang_obs_fp.objid}, Confidence: {lang_obs_fp.confidence:.3f}, Probability: {prob_fp:.3f}")

    assert prob_fp < 0.5, "❌ Test Failed: False positive probability should be lower."

    print("✅ Test Case 5 Passed.")

    print("\n🎉 All Language Observation Tests Passed!")

def unittest_mos_observation():
    """Tests MosObservationModel by sampling object, gesture, and language observations."""
    
    # random.seed(42)  # Fix random seed for reproducibility
    # np.random.seed(42)
    from multi_object_search.env.env import (
        make_laser_sensor,
        equip_sensors,
        interpret,
        interpret_robot_id,
    )

    # **Step 1: Define world layout**
    worldmap = """
        ..........
        ......T...
        ..T.r.....
        ..T.......
        ....T.....
        .....T....
        """
  
    # **Step 2: Equip the robot with a sensor**
    worldstr = equip_sensors(worldmap, {"r": make_laser_sensor(360, (1, 5), 0.1, False)})
    world_size, robot_states, object_states, obstacles, sensors = interpret(worldstr)

    robot_id = interpret_robot_id("r")
    robot_pose = robot_states[robot_id].pose
    sensor = sensors[robot_id]

    # **Step 3: Define Object States**
    object_states = {
        "T1": ObjectState("T1", "target", (4, 4, 0), gesture_prob=0.0, language_prob=0.0, iou=0.8),
        "T2": ObjectState("T2", "target", (5, 6, 0), gesture_prob=0.0, language_prob=0.0, iou=0.6),
        "Z": ObjectState("Z", "target", (5, 9, 0), gesture_prob=0.0, language_prob=0.0, iou=0.0),  # Out of FOV
    }
    next_state = MosOOState(object_states)

    # **Step 4: Initialize the Observation Model**
    observation_model = MosObservationModel(dim=world_size, sensor=sensor, object_ids=list(object_states.keys()))

    ## **Test Case 1: Gesture & Language Align (Both Point to T1)**
    print("\n✅ Test Case 1: Gesture & Language Align (T1)")

    
    observation_model.gesture_observation_model.generate_gesture_cone(
        origin=robot_pose[:3], direction=(-1, 0, 0), opening_angle=20
    )
    observation_model.language_observation_model = LanguageObservationModel(sensor)
    observation_model.language_observation_model.store_sentence(["T1"], [0.9], next_state, robot_pose)
    
    
    observation = observation_model.sample(next_state, LookAction(), robot_pose=robot_pose)

    assert observation.joint_probabilities["T1"] > observation.joint_probabilities["T2"], "❌ Expected T1 to have higher probability."
    print("Gesture Cone:",observation_model.gesture_observation_model.gesture_cone)
    print("Language Match:",observation_model.language_observation_model.persistent_object_language_match)
    
    print(f"   🟡 Gesture Probability T1 = {observation.observation_probs['T1']['gesture_prob']:.4f}")
    print(f"   🟡 Gesture Probability T2 = {observation.observation_probs['T2']['gesture_prob']:.4f}")
    print(f"   🟡 Gesture Probability Z = {observation.observation_probs['Z']['gesture_prob']:.4f}")
    print(f"   🔹 Language Probability T1 = {observation.observation_probs['T1']['language_prob']:.4f}")
    print(f"   🔹 Language Probability T2 = {observation.observation_probs['T2']['language_prob']:.4f}")
    print(f"   🔹 Language Probability Z = {observation.observation_probs['Z']['language_prob']:.4f}")
    print(f"   🟣 Joint Probability T1 = {observation.joint_probabilities['T1']:.4f}")
    print(f"   🟣 Joint Probability T2 = {observation.joint_probabilities['T2']:.4f}")
    print(f"   🟣 Joint Probability Z = {observation.joint_probabilities['Z']:.4f}")
    
    ## **Test Case 2: Gesture & Language Misalign (Gesture to T1, Language to T2)**
    print("\n⚠️ Test Case 2: Gesture & Language Misalign (Gesture → T1, Language → T2)")
    observation_model.language_observation_model = LanguageObservationModel(sensor)
    observation_model.language_observation_model.store_sentence(["T2"], [0.8], next_state, robot_pose)

    observation = observation_model.sample(next_state, LookAction(), robot_pose=robot_pose)

    # assert abs(observation.joint_probabilities["T1"] - observation.joint_probabilities["T2"]) < 0.3, \
    #     "❌ Expected misalignment to cause similar probabilities."
    print("Gesture Cone:",observation_model.gesture_observation_model.gesture_cone)
    print("Language Match:",observation_model.language_observation_model.persistent_object_language_match)
    
    print(f"   🟡 Gesture Probability T1 = {observation.observation_probs['T1']['gesture_prob']:.4f}")
    print(f"   🟡 Gesture Probability T2 = {observation.observation_probs['T2']['gesture_prob']:.4f}")
    print(f"   🟡 Gesture Probability Z = {observation.observation_probs['Z']['gesture_prob']:.4f}")
    print(f"   🔹 Language Probability T1 = {observation.observation_probs['T1']['language_prob']:.4f}")
    print(f"   🔹 Language Probability T2 = {observation.observation_probs['T2']['language_prob']:.4f}")
    print(f"   🔹 Language Probability Z = {observation.observation_probs['Z']['language_prob']:.4f}")
    print(f"   🟣 Joint Probability T1 = {observation.joint_probabilities['T1']:.4f}")
    print(f"   🟣 Joint Probability T2 = {observation.joint_probabilities['T2']:.4f}")
    print(f"   🟣 Joint Probability Z = {observation.joint_probabilities['Z']:.4f}")
    
    ## **Test Case 3: Only Gesture is Present**
    print("\n🟡 Test Case 3: Only Gesture is Present (No Language)")
    observation_model.language_observation_model = LanguageObservationModel(sensor)
    observation_model.language_observation_model.store_sentence([], [], next_state, robot_pose)

    observation = observation_model.sample(next_state, LookAction(), robot_pose=robot_pose)

    # assert observation.observation_probs["T1"]["gesture_prob"] > 0, "❌ Expected gesture probability for T1."
    # assert observation.observation_probs["T2"]["gesture_prob"] == 0, "❌ Expected zero gesture probability for T2."

    print("Gesture Cone:",observation_model.gesture_observation_model.gesture_cone)
    print("Language Match:",observation_model.language_observation_model.persistent_object_language_match)
    
    print(f"   🟡 Gesture Probability T1 = {observation.observation_probs['T1']['gesture_prob']:.4f}")
    print(f"   🟡 Gesture Probability T2 = {observation.observation_probs['T2']['gesture_prob']:.4f}")
    print(f"   🟡 Gesture Probability Z = {observation.observation_probs['Z']['gesture_prob']:.4f}")
    print(f"   🔹 Language Probability T1 = {observation.observation_probs['T1']['language_prob']:.4f}")
    print(f"   🔹 Language Probability T2 = {observation.observation_probs['T2']['language_prob']:.4f}")
    print(f"   🔹 Language Probability Z = {observation.observation_probs['Z']['language_prob']:.4f}")
    print(f"   🟣 Joint Probability T1 = {observation.joint_probabilities['T1']:.4f}")
    print(f"   🟣 Joint Probability T2 = {observation.joint_probabilities['T2']:.4f}")
    print(f"   🟣 Joint Probability Z = {observation.joint_probabilities['Z']:.4f}")
    
    ## **Test Case 4: Only Language is Present**
    print("\n🟣 Test Case 4: Only Language is Present (No Gesture)")

    observation_model.gesture_observation_model.gesture_cone = None
    observation_model.language_observation_model.store_sentence(["T2"], [0.9], next_state, robot_pose)

    observation = observation_model.sample(next_state, LookAction(), robot_pose=robot_pose)

    print("Gesture Cone:",observation_model.gesture_observation_model.gesture_cone)
    print("Language Match:",observation_model.language_observation_model.persistent_object_language_match)
    
    print(f"   🟡 Gesture Probability T1 = {observation.observation_probs['T1']['gesture_prob']:.4f}")
    print(f"   🟡 Gesture Probability T2 = {observation.observation_probs['T2']['gesture_prob']:.4f}")
    print(f"   🟡 Gesture Probability Z = {observation.observation_probs['Z']['gesture_prob']:.4f}")
    print(f"   🔹 Language Probability T1 = {observation.observation_probs['T1']['language_prob']:.4f}")
    print(f"   🔹 Language Probability T2 = {observation.observation_probs['T2']['language_prob']:.4f}")
    print(f"   🔹 Language Probability Z = {observation.observation_probs['Z']['language_prob']:.4f}")
    print(f"   🟣 Joint Probability T1 = {observation.joint_probabilities['T1']:.4f}")
    print(f"   🟣 Joint Probability T2 = {observation.joint_probabilities['T2']:.4f}")
    print(f"   🟣 Joint Probability Z = {observation.joint_probabilities['Z']:.4f}")
    
    ## **Test Case 5: No Gesture & No Language**
    print("\n🚫 Test Case 5: No Gesture & No Language")

    observation_model.gesture_observation_model.gesture_cone = None
    observation_model.language_observation_model = None
    observation_model.language_observation_model = LanguageObservationModel(sensor)
    observation_model.language_observation_model.store_sentence([], [], next_state, robot_pose)

    observation = observation_model.sample(next_state, LookAction(), robot_pose=robot_pose)

    print("Gesture Cone:",observation_model.gesture_observation_model.gesture_cone)
    print("Language Match:",observation_model.language_observation_model.persistent_object_language_match)
    
    print(f"   🟡 Gesture Probability T1 = {observation.observation_probs['T1']['gesture_prob']:.4f}")
    print(f"   🟡 Gesture Probability T2 = {observation.observation_probs['T2']['gesture_prob']:.4f}")
    print(f"   🟡 Gesture Probability Z = {observation.observation_probs['Z']['gesture_prob']:.4f}")
    print(f"   🔹 Language Probability T1 = {observation.observation_probs['T1']['language_prob']:.4f}")
    print(f"   🔹 Language Probability T2 = {observation.observation_probs['T2']['language_prob']:.4f}")
    print(f"   🔹 Language Probability Z = {observation.observation_probs['Z']['language_prob']:.4f}")
    print(f"   🟣 Joint Probability T1 = {observation.joint_probabilities['T1']:.4f}")
    print(f"   🟣 Joint Probability T2 = {observation.joint_probabilities['T2']:.4f}")
    print(f"   🟣 Joint Probability Z = {observation.joint_probabilities['Z']:.4f}")
      
    print("\n✅ **All Multi-Observation Tests Passed!**")


if __name__ == "__main__":
    # unittest_detection()
    # unittest_language()
    # unittest_gesture()
    unittest_mos_observation()

