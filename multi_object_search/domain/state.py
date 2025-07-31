"""Defines the State for the 2D Multi-Object Search domain;

Origin: Multi-Object Search using Object-Oriented POMDPs (ICRA 2019)
(extensions: action space changes, different sensor model, gridworld instead of
topological graph)

Description: Multi-Object Search in a 2D grid world.

State space:

    :math:`S_1 \\times S_2 \\times ... S_n \\times S_r`
    where :math:`S_i (1\leq i\leq n)` is the object state, with attribute
    "pose" :math:`(x,y)` and Sr is the state of the robot, with attribute
    "pose" :math:`(x,y)` and "objects_found" (set).
"""

import pomdp_py
import math


###### States ######
class ObjectState(pomdp_py.ObjectState):
    def __init__(self, objid, objclass, pose, 
                 gesture_prob=0.01, language_prob=0.01, iou=0.01):
        """Each object now stores:
        - `gesture_prob`: Probability that a human gesture is pointing to this object.
        - `language_prob`: Probability that language instruction refers to this object.
        - `iou`: IoU confidence score from segmentation.
        """
        if objclass not in {"obstacle", "target"}:
            raise ValueError(
                "Only allow object class to be either 'target' or 'obstacle'. Got %s"
                % objclass
            )
        if len(pose) == 3:  # If z is not provided, assume z=0 for 2D search
            pose = (pose[0], pose[1])

        super().__init__(objclass, {
            "pose": pose,
            "id": objid,
            "gesture_prob": gesture_prob,
            "language_prob": language_prob,
            "iou": iou
        })

    def __str__(self):
        return "ObjectState(%s, pose=%s, gesture_prob=%.3f, language_prob=%.3f, IoU=%.3f)" % (
            self.objclass, self.pose, 
            self.gesture_prob, self.language_prob, self.iou)

    @property
    def pose(self):
        return self.attributes["pose"]

    @property
    def objid(self):
        return self.attributes["id"]

    @property
    def gesture_prob(self):
        return self.attributes["gesture_prob"]

    @property
    def language_prob(self):
        return self.attributes["language_prob"]

    @property
    def iou(self):
        """Returns the IoU confidence score from segmentation."""
        return self.attributes["iou"]

    def update_gesture_prob(self, new_prob):
        """Update the gesture probability based on new observations."""
        self.attributes["gesture_prob"] = new_prob

    def update_language_prob(self, new_prob):
        """Update the language probability based on new observations."""
        self.attributes["language_prob"] = new_prob

    def update_iou(self, new_iou):
        """Update IoU confidence score."""
        self.attributes["iou"] = new_iou

class RobotState(pomdp_py.ObjectState):
    def __init__(self, robot_id, pose, objects_found, camera_direction):
        """Note: camera_direction is None unless the robot is looking at a direction,
        in which case camera_direction is the string e.g. look+x, or 'look'"""
        super().__init__(
            "robot",
            {
                "id": robot_id,
                "pose": pose,  # x,y,th
                "objects_found": objects_found,
                "camera_direction": camera_direction,
            },
        )

    def __str__(self):
        return "RobotState(%s,%s|%s)" % (
            str(self.objclass),
            str(self.pose),
            str(self.objects_found),
        )

    def __repr__(self):
        return str(self)

    @property
    def pose(self):
        return self.attributes["pose"]

    @property
    def robot_pose(self):
        return self.attributes["pose"]

    @property
    def objects_found(self):
        return self.attributes["objects_found"]


class MosOOState(pomdp_py.OOState):
    
    def __init__(self, object_states):
        super().__init__(object_states)

    def object_pose(self, objid):
        # return 3d object pose instead of 2d
        return self.object_states[objid].pose

    def pose(self, objid):
        return self.object_pose(objid)

    @property
    def object_poses(self):
        return {objid: self.object_states[objid].pose for objid in self.object_states}

    @property
    def gesture_probs(self):
        """Returns a dictionary mapping object IDs to gesture probabilities."""
        return {objid: self.object_states[objid].gesture_prob 
                if "gesture_prob" in self.object_states[objid].attributes else 0.0
                for objid in self.object_states}

    @property
    def language_probs(self):
        """Returns a dictionary mapping object IDs to language probabilities."""
        return {objid: self.object_states[objid].language_prob 
                if "language_prob" in self.object_states[objid].attributes else 0.0
                for objid in self.object_states}

    @property
    def iou_scores(self):
        """Returns a dictionary mapping object IDs to their IoU confidence scores."""
        return {objid: self.object_states[objid].iou 
                if "iou" in self.object_states[objid].attributes else 0.0
                for objid in self.object_states}

    def update_gesture_prob(self, objid, new_prob):
        """Update gesture probability for a specific object."""
        if objid in self.object_states:
            self.object_states[objid].update_gesture_prob(new_prob)

    def update_language_prob(self, objid, new_prob):
        """Update language probability for a specific object."""
        if objid in self.object_states:
            self.object_states[objid].update_language_prob(new_prob)

    def update_iou(self, objid, new_iou):
        """Update IoU confidence score for a specific object."""
        if objid in self.object_states:
            self.object_states[objid].update_iou(new_iou)
            
    def __str__(self):
        return "MosOOState%s" % (str(self.object_states))

    def __repr__(self):
        return str(self)
