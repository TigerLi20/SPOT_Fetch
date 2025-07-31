import pomdp_py
import json
import numpy as np
import time
from pose_detection.gesture_util import read_json_locked

# 🔹 File path to live detection data
file_path = ".tmp/detection_confidence.json"

# 🔹 Function to read and update belief dynamically
def get_dynamic_observations():
    data = read_json_locked(file_path)
    return {mark["mark"]: mark["combined_prob"] for mark in data}  # Extract marks & probs

# 🔹 Define a Dynamic Belief State (Instead of Predefined World States)
class BeliefState(pomdp_py.State):
    def __init__(self, belief):
        self.belief = belief  # Belief is over detected objects

    def __repr__(self):
        return f"Belief({self.belief})"

# 🔹 Define Search Actions (Move to High-Confidence Objects)
class MoveToObject(pomdp_py.Action):
    def __init__(self, mark_id):
        self.mark_id = mark_id  # Move toward this mark

    def __repr__(self):
        return f"MoveToObject({self.mark_id})"

# 🔹 Observation Model: Uses `combined_prob` Directly
class DynamicObservationModel(pomdp_py.ObservationModel):
    def probability(self, observation, next_state, action):
        return observation.get(action.mark_id, 0.01)  # Default low prob if missing

# 🔹 Transition Model: No Predefined Model, Just Identity Transition
class IdentityTransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):
        return 1.0  # Assume action is successful, since no world model

# 🔹 Reward Model: Reward for Selecting the Right Object
class SearchRewardModel(pomdp_py.RewardModel):
    def reward(self, state, action, next_state):
        max_mark = max(state.belief, key=state.belief.get)  # Object with highest belief
        return 1 if action.mark_id == max_mark else -1  # Reward if action selects max

# 🔹 Define the Search Agent with Dynamic Belief Updates
class SearchAgent(pomdp_py.Agent):
    def __init__(self):
        initial_belief = get_dynamic_observations()  # Initialize belief
        belief = pomdp_py.Histogram(initial_belief)  # Convert to POMDP belief
        policy_model = pomdp_py.RandomRollout()
        transition_model = IdentityTransitionModel()
        observation_model = DynamicObservationModel()
        reward_model = SearchRewardModel()
        super().__init__(belief, policy_model, transition_model, observation_model, reward_model)

    def update_belief(self):
        new_observations = get_dynamic_observations()
        for obj_id, prob in new_observations.items():
            self.belief[obj_id] = prob  # Update belief dynamically

# 🔹 Define the Search POMDP
class DynamicSearchPOMDP(pomdp_py.POMDP):
    def __init__(self):
        agent = SearchAgent()
        env = pomdp_py.Environment(BeliefState(agent.belief))
        super().__init__(agent, env, discount_factor=0.95)

# 🔹 Run the POMDP
pomdp = DynamicSearchPOMDP()

# 🔹 Search Simulation Loop
for _ in range(10):
    pomdp.agent.update_belief()  # Update belief dynamically
    best_object = max(pomdp.agent.belief, key=pomdp.agent.belief.get)
    action = MoveToObject(best_object)  # Move towards highest probability object
    pomdp.step(action)  # Perform action
    print(f"Action: {action}, Updated Belief: {pomdp.agent.belief}")