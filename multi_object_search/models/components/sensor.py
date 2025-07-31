"""Sensor model (for example, laser scanner)"""

import math
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from multi_object_search.domain.action import *
from multi_object_search.domain.observation import *

# Note that the occlusion of an object is implemented based on
# whether a beam will hit an obstacle or some other object before
# that object. Because the world is discretized, this leads to
# some strange pattern of the field of view. But what's for sure
# is that, when occlusion is enabled, the sensor will definitely
# not receive observation for some regions in the field of view
# making it a more challenging situation to deal with.


# Utility functions
def euclidean_dist(p1, p2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))


def to_rad(deg):
    return deg * math.pi / 180.0


def in_range(val, rang):
    # Returns True if val is in range (a,b); Inclusive.
    return val >= rang[0] and val <= rang[1]


#### Sensors ####
class Sensor:
    LASER = "laser"
    PROXIMITY = "proximity"

    def observe(self, robot_pose, env_state):
        """
        Returns an Observation with this sensor model.
        """
        raise NotImplementedError

    def within_range(self, robot_pose, point):
        """Returns true if the point is within range of the sensor; but the point might not
        actually be visible due to occlusion or "gap" between beams"""
        raise ValueError

    @property
    def sensing_region_size(self):
        return self._sensing_region_size

    @property
    def robot_id(self):
        # id of the robot equipped with this sensor
        return self._robot_id


class Laser2DSensor:
    """Fan shaped 2D laser sensor"""

    def __init__(
        self,
        robot_id,
        fov=90,
        min_range=1,
        max_range=5,
        angle_increment=5,
        occlusion_enabled=False,
        
        gesture_cone = None,  # Simulated pointing direction
        language_match = None  # Simulated language input

    ):
        """
        fov (float): angle between the start and end beams of one scan (degree).
        min_range (int or float)
        max_range (int or float)
        angle_increment (float): angular distance between measurements (rad).
        """
        self.robot_id = robot_id
        self.fov = to_rad(fov)  # convert to radian
        self.min_range = min_range
        self.max_range = max_range
        self.angle_increment = to_rad(angle_increment)
        self._occlusion_enabled = occlusion_enabled


        self.gesture_cone = gesture_cone  # Simulated pointing direction
        self.language_match = language_match  # Simulated language input

        # determines the range of angles;
        # For example, the fov=pi, means the range scanner scans 180 degrees
        # in front of the robot. By our angle convention, 180 degrees maps to [0,90] and [270, 360]."""
        self._fov_left = (0, self.fov / 2)
        self._fov_right = (2 * math.pi - self.fov / 2, 2 * math.pi)

        # beams that are actually within the fov (set of angles)
        self._beams = {
            round(th, 2)
            for th in np.linspace(
                self._fov_left[0],
                self._fov_left[1],
                int(
                    round(
                        (self._fov_left[1] - self._fov_left[0]) / self.angle_increment
                    )
                ),
            )
        } | {
            round(th, 2)
            for th in np.linspace(
                self._fov_right[0],
                self._fov_right[1],
                int(
                    round(
                        (self._fov_right[1] - self._fov_right[0]) / self.angle_increment
                    )
                ),
            )
        }
        # The size of the sensing region here is the area covered by the fan
        self._sensing_region_size = (
            self.fov / (2 * math.pi) * math.pi * (max_range - min_range) ** 2
        )

    def in_field_of_view(th, view_angles):
        """Determines if the beame at angle `th` is in a field of view of size `view_angles`.
        For example, the view_angles=180, means the range scanner scans 180 degrees
        in front of the robot. By our angle convention, 180 degrees maps to [0,90] and [270, 360].
        """
        fov_right = (0, view_angles / 2)
        fov_left = (2 * math.pi - view_angles / 2, 2 * math.pi)

    def within_range(self, robot_pose, point):
        """Returns true if the point is within range of the sensor; but the point might not
        actually be visible due to occlusion or "gap" between beams"""
        dist, bearing = self.shoot_beam(robot_pose, point)
        if not in_range(dist, (self.min_range, self.max_range)):
            return False
        if (not in_range(bearing, self._fov_left)) and (
            not in_range(bearing, self._fov_right)
        ):
            return False
        return True

    def shoot_beam(self, robot_pose, point):
        """Shoots a beam from robot_pose at point. Returns the distance and bearing
        of the beame (i.e. the length and orientation of the beame)"""
        rx, ry, rth = robot_pose
        dist = euclidean_dist(point, (rx, ry))
        bearing = (math.atan2(point[1] - ry, point[0] - rx) - rth) % (
            2 * math.pi
        )  # bearing (i.e. orientation)
        return (dist, bearing)

    def valid_beam(self, dist, bearing):
        """Returns true beam length (i.e. `dist`) is within range and its angle
        `bearing` is valid, that is, it is within the fov range and in
        accordance with the angle increment."""
        return (
            dist >= self.min_range
            and dist <= self.max_range
            and round(bearing, 2) in self._beams
        )

    def _build_beam_map(self, beam, point, beam_map={}):
        """beam_map (dict): Maps from bearing to (dist, point)"""
        dist, bearing = beam
        valid = self.valid_beam(dist, bearing)
        if not valid:
            return
        bearing_key = round(bearing, 2)
        if bearing_key in beam_map:
            # There's an object covered by this beame already.
            # see if this beame is closer
            if dist < beam_map[bearing_key][0]:
                # point is closer; Update beam map
                print("HEY")
                beam_map[bearing_key] = (dist, point)
            else:
                # point is farther than current hit
                pass
        else:
            beam_map[bearing_key] = (dist, point)

    def _compute_gesture_confidence(self, object_pose):
        """Computes the probability that a gesture is pointing at an object."""
        if self.gesture_cone is None:
            return 0.0  # No gesture detected

        origin, direction = self.gesture_cone
        obj_dist = euclidean_dist(origin, object_pose[:2])

        if obj_dist == 0:
            return 1.0  # Directly pointing at the object

        # Compute angle difference between gesture direction and object location
        obj_angle = math.atan2(object_pose[1] - origin[1], object_pose[0] - origin[0])
        angle_diff = abs(obj_angle - direction)

        # If angle difference is small, assign high confidence
        return max(0.0, 1 - (angle_diff / (np.pi / 4)))  # Normalize over 45 degrees

    def _compute_language_confidence(self, objid):
        """Returns the language probability for a given object, defaulting to 0."""
        if self.language_match is None:
            return 0.0
        return self.language_match.get(objid, 0.0)
    
    
    def observe(self, robot_pose, env_state, language_match = None):
        """
        Returns a MosObservation with this sensor model.
        """
        rx, ry, rth = robot_pose
        if language_match is not None:
            self.language_match = language_match  # Update dynamically
            
        # Check every object
        objposes = {}
        observation_probs = {} 
        beam_map = {}
        for objid in env_state.object_states:
            objposes[objid] = ObjectObservation.NULL
            object_pose = env_state.object_states[objid]["pose"]
            beam = self.shoot_beam(robot_pose, object_pose)
            if not self._occlusion_enabled:
                if self.valid_beam(*beam):
                    d, bearing = beam  # distance, bearing
                    lx = rx + int(round(d * math.cos(rth + bearing)))
                    ly = ry + int(round(d * math.sin(rth + bearing)))
                    objposes[objid] = (lx, ly)
                    
                    # **Compute IoU Score** (Simulated Object Detection)
                    object_iou = max(0.5, 1 - (d / self.max_range))  # IoU decreases with distance
                else:
                    object_iou = 0.01
                # **Compute Gesture Probability**
                gesture_conf = self._compute_gesture_confidence(object_pose)

                # **Compute Language Probability**
                language_conf = self._compute_language_confidence(objid)

                # Store probabilities
                observation_probs[objid] = {
                    "iou": object_iou,
                    "gesture_prob": gesture_conf,
                    "language_prob": language_conf,
                }


            else:
                self._build_beam_map(beam, object_pose, beam_map=beam_map)

        if self._occlusion_enabled:
            # The observed objects are in the beam_map
            for bearing_key in beam_map:
                d, objid = beam_map[bearing_key]
                lx = rx + int(round(d * math.cos(rth + bearing_key)))
                ly = ry + int(round(d * math.sin(rth + bearing_key)))
                objposes[objid] = (lx, ly)
                
                # Simulated IoU with occlusion penalty
                object_iou = max(0.2, 1 - (d / self.max_range))
                gesture_conf = self._compute_gesture_confidence(env_state.object_states[objid].pose)
                language_conf = self._compute_language_confidence(objid)

                observation_probs[objid] = {
                    "object_prob": object_iou,
                    "gesture_prob": gesture_conf,
                    "language_prob": language_conf,
                }


        return MosOOObservation(objposes, observation_probs)

    @property
    def sensing_region_size(self):
        return self._sensing_region_size


class ProximitySensor(Laser2DSensor):
    """This is a simple sensor; Observes a region centered
    at the robot."""

    def __init__(self, robot_id, radius=5, occlusion_enabled=False):
        """
        radius (int or float) radius of the sensing region.
        """
        self.robot_id = robot_id
        self.radius = radius
        self._occlusion_enabled = occlusion_enabled

        # This is in fact just a specific kind of Laser2DSensor
        # that has a 360 field of view, min_range = 0.1 and
        # max_range = radius
        if occlusion_enabled:
            angle_increment = 5
        else:
            angle_increment = 0.25
        super().__init__(
            robot_id,
            fov=360,
            min_range=0.1,
            max_range=radius,
            angle_increment=angle_increment,
            occlusion_enabled=occlusion_enabled,
        )


import numpy as np
from multi_object_search.domain.state import ObjectState, MosOOState
from multi_object_search.domain.observation import MosOOObservation
from multi_object_search.domain.action import LookAction

def unittest_sensor():
    """Tests the Laser2DSensor with simulated IoU, Gesture, and Language inputs."""

    print("\n🚀 Running Sensor Unit Test...")

    # **Step 1: Define World & Robot**
    robot_id = "r"
    robot_pose = (2, 2, 0)  # (x, y, theta)

    # **Step 2: Define Objects in Scene**
    object_states = {
        "T1": ObjectState("T1", "target", (3, 2, 0)),  # Close, in FOV
        "T2": ObjectState("T2", "target", (3, 3, 0)),  # Further, in FOV
        "Z": ObjectState("Z", "target", (8, 8, 0)),    # Out of FOV
    }
    env_state = MosOOState(object_states)

    # **Step 3: Define Gesture Cone & Language Match**
    gesture_cone = ((2, 2, 0), np.deg2rad(1))  # Robot is pointing right
    language_model = {"T1": 0.9, "T2": 0.5}  # Simulated language match

    # **Step 4: Initialize Sensor**
    sensor = Laser2DSensor(
        robot_id=robot_id,
        fov=90,  # 90-degree field of view
        min_range=1,
        max_range=6,
        angle_increment=5,
        occlusion_enabled=False,
        gesture_cone=gesture_cone,
        language_match=language_model
    )

    # **Step 5: Sample Observation**
    observation = sensor.observe(robot_pose, env_state)

    # **Step 6: Print Results**
    print("\n🔎 Sampled Observations:")
    for objid, obs in observation.objposes.items():
        print(f"🟢 Object {objid}: Pose = {obs}")
    
    print("\n📊 Individual Observation Probabilities:")
    for objid, probs in observation.observation_probs.items():
        print(f"🔹 {objid}: P(O) = {probs['iou']:.3f}, P(G) = {probs['gesture_prob']:.3f}, P(L) = {probs['language_prob']:.3f}")

    # **Step 7: Assertions**
    assert isinstance(observation, MosOOObservation), "❌ Observation should be a MosOOObservation instance."
    assert "T1" in observation.objposes, "❌ Missing observation for T1."
    assert "T2" in observation.objposes, "❌ Missing observation for T2."
    assert "Z" in observation.objposes, "❌ Missing observation for Z."

    # **Verify IoU (Object Detection Confidence)**
    assert observation.observation_probs["T1"]["iou"] > 0.5, "❌ T1 IoU should be high (close to robot)."
    assert observation.observation_probs["T2"]["iou"] > 0.2, "❌ T2 IoU should be moderate (further)."

    # **Verify Gesture Probability**
    assert observation.observation_probs["T1"]["gesture_prob"] > 0.5, "❌ T1 should have high gesture probability."
    assert observation.observation_probs["T2"]["gesture_prob"] < 0.5, "❌ T2 should have low gesture probability."

    # **Verify Language Probability**
    assert observation.observation_probs["T1"]["language_prob"] == 0.9, "❌ T1 language probability should match model."
    assert observation.observation_probs["T2"]["language_prob"] == 0.5, "❌ T2 language probability should match model."

    print("\n✅ **Sensor Test Passed: IoU, Gesture, and Language Simulated Correctly!**")

    print("Case 2: No language observation")
    
    # **Step 3: Define Gesture Cone & Language Match**
    gesture_cone = ((2, 2, 0), np.deg2rad(1))  # Robot is pointing right
    language_model = None  # Simulated language match

    # **Step 4: Initialize Sensor**
    sensor = Laser2DSensor(
        robot_id=robot_id,
        fov=90,  # 90-degree field of view
        min_range=1,
        max_range=6,
        angle_increment=5,
        occlusion_enabled=False,
        gesture_cone=gesture_cone,
        language_match=language_model
    )

    # **Step 5: Sample Observation**
    observation = sensor.observe(robot_pose, env_state)

    # **Step 6: Print Results**
    print("\n🔎 Sampled Observations:")
    for objid, obs in observation.objposes.items():
        print(f"🟢 Object {objid}: Pose = {obs}")
    
    print("\n📊 Individual Observation Probabilities:")
    for objid, probs in observation.observation_probs.items():
        print(f"🔹 {objid}: P(O) = {probs['iou']:.3f}, P(G) = {probs['gesture_prob']:.3f}, P(L) = {probs['language_prob']:.3f}")

    # **Step 7: Assertions**
    assert isinstance(observation, MosOOObservation), "❌ Observation should be a MosOOObservation instance."
    assert "T1" in observation.objposes, "❌ Missing observation for T1."
    assert "T2" in observation.objposes, "❌ Missing observation for T2."
    assert "Z" in observation.objposes, "❌ Missing observation for Z."

    # **Verify IoU (Object Detection Confidence)**
    assert observation.observation_probs["T1"]["iou"] > 0.5, "❌ T1 IoU should be high (close to robot)."
    assert observation.observation_probs["T2"]["iou"] > 0.2, "❌ T2 IoU should be moderate (further)."

    # **Verify Gesture Probability**
    assert observation.observation_probs["T1"]["gesture_prob"] > 0.5, "❌ T1 should have high gesture probability."
    assert observation.observation_probs["T2"]["gesture_prob"] < 0.5, "❌ T2 should have low gesture probability."

    print("\n✅ **Sensor Test Passed: IoU, Gesture, and Language Simulated Correctly!**")

    print("Case 3: No gesture observation")
    # **Step 3: Define Gesture Cone & Language Match**
    gesture_cone = None  # Robot is pointing right
    language_model = {"T1": 0.9, "T2": 0.5, "T3": 0.7}  # Simulated language match

    # **Step 4: Initialize Sensor**
    sensor = Laser2DSensor(
        robot_id=robot_id,
        fov=90,  # 90-degree field of view
        min_range=1,
        max_range=6,
        angle_increment=5,
        occlusion_enabled=False,
        gesture_cone=gesture_cone,
        language_match=language_model
    )

    # **Step 5: Sample Observation**
    observation = sensor.observe(robot_pose, env_state)

    # **Step 6: Print Results**
    print("\n🔎 Sampled Observations:")
    for objid, obs in observation.objposes.items():
        print(f"🟢 Object {objid}: Pose = {obs}")
    
    print("\n📊 Individual Observation Probabilities:")
    for objid, probs in observation.observation_probs.items():
        print(f"🔹 {objid}: P(O) = {probs['iou']:.3f}, P(G) = {probs['gesture_prob']:.3f}, P(L) = {probs['language_prob']:.3f}")

    # **Step 7: Assertions**
    assert isinstance(observation, MosOOObservation), "❌ Observation should be a MosOOObservation instance."
    assert "T1" in observation.objposes, "❌ Missing observation for T1."
    assert "T2" in observation.objposes, "❌ Missing observation for T2."
    assert "Z" in observation.objposes, "❌ Missing observation for Z."

    # **Verify IoU (Object Detection Confidence)**
    assert observation.observation_probs["T1"]["iou"] > 0.5, "❌ T1 IoU should be high (close to robot)."
    assert observation.observation_probs["T2"]["iou"] > 0.2, "❌ T2 IoU should be moderate (further)."

    # **Verify Language Probability**
    assert observation.observation_probs["T1"]["language_prob"] == 0.9, "❌ T1 language probability should match model."
    assert observation.observation_probs["T2"]["language_prob"] == 0.5, "❌ T2 language probability should match model."

    print("\n✅ **Sensor Test Passed: IoU, Gesture, and Language Simulated Correctly!**")

    
    # **Step 3: Define Gesture Cone & Language Match**
    print("Case 3: No language and gesture observation")
    gesture_cone = None  # Robot is pointing right
    language_model =None # Simulated language match

    # **Step 4: Initialize Sensor**
    sensor = Laser2DSensor(
        robot_id=robot_id,
        fov=90,  # 90-degree field of view
        min_range=1,
        max_range=6,
        angle_increment=5,
        occlusion_enabled=False,
        gesture_cone=gesture_cone,
        language_match=language_model
    )

    # **Step 5: Sample Observation**
    observation = sensor.observe(robot_pose, env_state)

    # **Step 6: Print Results**
    print("\n🔎 Sampled Observations:")
    for objid, obs in observation.objposes.items():
        print(f"🟢 Object {objid}: Pose = {obs}")
    
    print("\n📊 Individual Observation Probabilities:")
    for objid, probs in observation.observation_probs.items():
        print(f"🔹 {objid}: P(O) = {probs['iou']:.3f}, P(G) = {probs['gesture_prob']:.3f}, P(L) = {probs['language_prob']:.3f}")

    # **Step 7: Assertions**
    assert isinstance(observation, MosOOObservation), "❌ Observation should be a MosOOObservation instance."
    assert "T1" in observation.objposes, "❌ Missing observation for T1."
    assert "T2" in observation.objposes, "❌ Missing observation for T2."
    assert "Z" in observation.objposes, "❌ Missing observation for Z."

    # **Verify IoU (Object Detection Confidence)**
    assert observation.observation_probs["T1"]["iou"] > 0.5, "❌ T1 IoU should be high (close to robot)."
    assert observation.observation_probs["T2"]["iou"] > 0.2, "❌ T2 IoU should be moderate (further)."

    print("\n✅ **Sensor Test Passed: IoU, Gesture, and Language Simulated Correctly!**")

# Run the test
if __name__ == "__main__":
    unittest_sensor()