import pomdp_py

###### Observation ######
class ObjectObservation(pomdp_py.Observation):
    """The xy pose of the object is observed; or NULL if not observed.
       Also stores IoU confidence score for object detection."""

    NULL = None

    def __init__(self, objid, pose, iou=0.01):
        """
        objid: Object ID.
        pose: Tuple (x, y, z) for 3D position, or NULL if not observed.
        iou: IoU confidence score from segmentation.
        """
        self.objid = objid
        if pose == ObjectObservation.NULL:
            self.pose = pose
        elif type(pose) == tuple and (len(pose) == 3 or len(pose) == 2):
            if len(pose) == 2:  # If z is missing, assume z=0 for 2D cases
                pose = (pose[0], pose[1], 0)
            self.pose = pose
        else:
            raise ValueError("Invalid observation %s for object" % (str(pose), objid))
        self.iou = iou  # Store IoU confidence score

    def __hash__(self):
        return hash((self.objid, self.pose, self.iou))

    def __eq__(self, other):
        if not isinstance(other, ObjectObservation):
            return False
        return self.objid == other.objid and self.pose == other.pose and self.iou == other.iou

    def __str__(self):
        return f"ObjectObservation(objid={self.objid}, pose={self.pose}, iou={self.iou:.3f})"


class GestureObservation(pomdp_py.Observation):
    """Stores gesture observation in a structured format."""

    def __init__(self, objid, confidence):
        """
        objid: ID of the referenced object.
        confidence: Probability that the gesture is pointing at this object.
        """
        self.objid = objid
        self.confidence = confidence  # Gesture detection confidence

    def __hash__(self):
        return hash((self.objid, self.confidence))

    def __eq__(self, other):
        return isinstance(other, GestureObservation) and self.objid == other.objid and self.confidence == other.confidence

    def __str__(self):
        return f"GestureObservation(objid={self.objid}, confidence={self.confidence:.3f})"


class LanguageObservation(pomdp_py.Observation):
    """Stores the language-based observation, including the full sentence."""

    def __init__(self, objid, confidence=0):
        """
        objid: The ID of the object being referenced.
        sentence: The human language instruction as a string.
        """
        self.objid = objid  
        self.confidence = confidence

    def __hash__(self):
        return hash((self.objid, self.confidence))

    def __eq__(self, other):
        return (
            isinstance(other, LanguageObservation)
            and self.objid == other.objid
            and self.confidence == other.confidence
        )

    def __str__(self):
        return f"LanguageObservation(objid={self.objid}, confidence='{self.confidence}')"

    def __repr__(self):
        return str(self)

class MosOOObservation(pomdp_py.OOObservation):
    """Observation for Multi-Object Search, combining object, gesture, and language observations."""

    def __init__(self, objposes, observation_probs, joint_probabilities=None):
        """
        - objposes (dict): Maps objid to observed object poses.
        - observation_probs (dict): Stores P(O), P(G), P(L) before computing joint probability.
        - joint_probabilities (dict): Stores normalized joint probabilities.
        """
        self.objposes = objposes
        self.observation_probs = observation_probs  # Stores individual probabilities before merging
        self.joint_probabilities = joint_probabilities if joint_probabilities else {}

        # Ensure hashcode considers all attributes
        self._hashcode = hash((frozenset(self.objposes.items()),
                               frozenset(self.joint_probabilities.items())))

    def for_obj(self, objid):
        """Returns observation details for a specific object."""
        obs_probs = dict(self.observation_probs.get(objid, {}))  # Convert tuple back to dict for easier access
        return {
            "pose": self.objposes.get(objid, ObjectObservation.NULL),
            "object_prob": obs_probs.get("object_prob", 0.01),
            "gesture_prob": obs_probs.get("gesture_prob", 0.01),
            "language_prob": obs_probs.get("language_prob", 0.01),
            "joint_prob": self.joint_probabilities.get(objid, 0.01)
        }


    def __hash__(self):
        return self._hashcode

    def __eq__(self, other):
        return (self.objposes == other.objposes and
                self.observation_probs == other.observation_probs and
                self.joint_probabilities == other.joint_probabilities)

    def __str__(self):
        return (f"MosOOObservation(objects={self.objposes}, "
                f"observation_probs={self.observation_probs}, "
                f"joint_prob={self.joint_probabilities})")

    def __repr__(self):
        return str(self)

    def factor(self, next_state, *params, **kwargs):
        """Factor this OO-observation by objects"""
        return {
            objid: ObjectObservation(objid, self.objposes[objid])
            for objid in next_state.object_states
            if objid != next_state.robot_id
        }
        
        
    @classmethod
    def merge(cls, object_observations, gesture_observations, language_observations, next_state):
        """Merge object, gesture, and language observations into a single MosOOObservation."""
        observation_probs = {}
        objposes = {}

        # **Step 1: Extract and Normalize Individual Probabilities**
        total_obj_prob = sum(obj.iou for obj in object_observations.values())
        total_gest_prob = sum(gesture_observations[objid].confidence for objid in gesture_observations)
        total_lang_prob = sum(language_observations[objid].confidence for objid in language_observations)

        for objid, obj_obs in object_observations.items():
            obj_state = next_state.object_states[objid]

            # Normalize individual probabilities
            object_prob = obj_obs.iou / total_obj_prob if total_obj_prob > 0 else 0
            gesture_prob = gesture_observations.get(objid, GestureObservation(None, 0)).confidence / total_gest_prob if total_gest_prob > 0 else 0
            language_prob = language_observations.get(objid, LanguageObservation(None, 0)).confidence / total_lang_prob if total_lang_prob > 0 else 0

            # Store probabilities
            observation_probs[objid] = {
                "object_prob": object_prob,
                "gesture_prob": gesture_prob,
                "language_prob": language_prob
            }

            # Store observed object pose
            objposes[objid] = obj_obs.pose

        # **Step 2: Compute Normalized Joint Probability**
        joint_probabilities = {}
        joint_prob_sum = 0

        for objid, obs_probs in observation_probs.items():
            joint_prob = obs_probs["gesture_prob"] * obs_probs["language_prob"] * obs_probs["object_prob"]
            joint_probabilities[objid] = joint_prob
            joint_prob_sum += joint_prob

        # **Step 3: Normalize Joint Probabilities**
        if joint_prob_sum > 0:
            joint_probabilities = {objid: prob / joint_prob_sum for objid, prob in joint_probabilities.items()}
        else:
            num_objects = len(joint_probabilities)
            joint_probabilities = {objid: 1.0 / num_objects for objid in joint_probabilities} if num_objects > 0 else {}

        return MosOOObservation(objposes=objposes, observation_probs=observation_probs, joint_probabilities=joint_probabilities)
# class MosOOObservation(pomdp_py.OOObservation):
#     """Observation for Multi-Object Search, factoring objects, gesture, and language."""

#     def __init__(self, objposes, gesture=None, language=None):
#         """
#         objposes (dict): Maps objid to 2D pose or NULL.
#         gesture (dict): Maps objid to GestureObservation.
#         language (dict): Maps objid to LanguageObservation.
#         """
#         self.objposes = objposes
#         self.gesture = gesture if gesture else {}
#         self.language = language if language else {}

#         # Ensure hashcode considers all attributes
#         self._hashcode = hash((frozenset(self.objposes.items()),
#                                frozenset(self.gesture.items()),
#                                frozenset(self.language.items())))

#     def for_obj(self, objid):
#         """Returns observation components for a specific object."""
#         return {
#             "object": ObjectObservation(objid, self.objposes.get(objid, ObjectObservation.NULL)),
#             "gesture": self.gesture.get(objid, None),
#             "language": self.language.get(objid, None),
#         }

#     def __hash__(self):
#         return self._hashcode

#     def __eq__(self, other):
#         if not isinstance(other, MosOOObservation):
#             return False
#         return (self.objposes == other.objposes and
#                 self.gesture == other.gesture and
#                 self.language == other.language)

#     def __str__(self):
#         return (f"MosOOObservation(objects={self.objposes}, "
#                 f"gestures={self.gesture}, language={self.language})")

#     def __repr__(self):
#         return str(self)

#     def factor(self, next_state, *params, **kwargs):
#         """Factor this OO-observation by objects."""
#         return {
#             objid: ObjectObservation(objid, self.objposes[objid])
#             for objid in next_state.object_states
#             if objid != next_state.robot_id
#         }

#     @classmethod
#     def merge(cls, object_observations, gesture_observations, language_observations, next_state, *params, **kwargs):
#         """Merge object, gesture, and language observations into a single MosOOObservation.

#         Args:
#             object_observations (dict): Maps objid to ObjectObservation.
#             gesture_observations (dict): Maps objid to GestureObservation.
#             language_observations (dict): Maps objid to LanguageObservation.
#             next_state: The next state used for filtering.

#         Returns:
#             MosOOObservation: A merged observation object.
#         """
#         filtered_object_observations = {
#             objid: (obj.pose, obj.iou)  # Ensure both pose & IoU are stored
#             for objid, obj in object_observations.items()
#             if objid != next_state.robot_id  # Filter out robot ID before processing
#         }

#         return MosOOObservation(
#             objposes={
#                 objid: filtered_object_observations[objid][0]  # Store just pose
#                 for objid in filtered_object_observations
#             },
#             gesture={
#                 objid: gesture_observations.get(objid, None) for objid in filtered_object_observations
#             },
#             language={
#                 objid: language_observations.get(objid, None) for objid in filtered_object_observations
#             }
#         )