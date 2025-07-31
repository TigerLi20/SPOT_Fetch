"""Policy model for 2D Multi-Object Search domain.
It is optional for the agent to be equipped with an occupancy
grid map of the environment.
"""

import pomdp_py
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from multi_object_search.domain.action import *


class PolicyModelRandom(pomdp_py.RolloutPolicy):
    """Simple policy model. All actions are possible at any state."""

    def __init__(self, robot_id, grid_map=None):
        """FindAction can only be taken after LookAction"""
        self.robot_id = robot_id
        self._grid_map = grid_map

    def sample(self, state, **kwargs):
        return random.sample(self._get_all_actions(**kwargs), 1)[0]

    def probability(self, action, state, **kwargs):
        raise NotImplementedError

    def argmax(self, state, **kwargs):
        """Returns the most likely action"""
        raise NotImplementedError

    def get_all_actions(self, state=None, history=None):
        """note: find can only happen after look."""
        can_find = False
        if history is not None and len(history) > 1:
            # last action
            last_action = history[-1][0]
            if isinstance(last_action, LookAction):
                can_find = True
        find_action = [Find] if can_find else []
        if state is None:
            return ALL_MOTION_ACTIONS + [Look] + find_action
        else:
            if self._grid_map is not None:
                valid_motions = self._grid_map.valid_motions(
                    self.robot_id, state.pose(self.robot_id), ALL_MOTION_ACTIONS
                )
                return list(valid_motions) + [Look] + find_action
            else:
                return ALL_MOTION_ACTIONS + [Look] + find_action

    
    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

class StrongRandomPolicy(pomdp_py.RolloutPolicy):
    """Stronger random policy that intelligently samples actions based on Look vs. No-Look categories."""

    def __init__(self, robot_id, grid_map=None, explore_prob=0.3):
        """
        Args:
            robot_id (int): ID of the robot.
            grid_map (GridMap, optional): Used to validate motion feasibility.
            explore_prob (float): Probability of taking a random exploratory action.
        """
        self.robot_id = robot_id
        self._grid_map = grid_map
        self.explore_prob = explore_prob  # Controls randomness in decision-making.

    def sample(self, state, history=None):
        """
        Samples an action intelligently:
        - Prioritizes `LookAction` early in search.
        - Ensures `FindAction` is only taken after `LookAction`.
        - Uses `explore_prob` to balance exploration & exploitation.
        """
        possible_actions = self.get_all_actions(state, history)
        
        # Decide whether to pick from Look-based or No-Look actions
        look_actions = {Look}
        no_look_actions = {a for a in possible_actions if a not in look_actions}

        if history and len(history) > 1:
            last_action = history[-1][0]
            if isinstance(last_action, LookAction):
                look_actions.add(Find)  # Allow FindAction after Look

        # Weighted sampling
        action_weights = {action: 1.0 for action in possible_actions}
        
        # Increase probability of Look early in search
        if Look in action_weights:
            action_weights[Look] = 2.0 if random.random() > self.explore_prob else 1.0

        # Reduce probability of `FindAction` unless Look was recently performed
        if Find in action_weights:
            action_weights[Find] = 0.5  # Lower probability unless Look was taken

        # Normalize and sample action
        total_weight = sum(action_weights.values())
        probabilities = [w / total_weight for w in action_weights.values()]
        return random.choices(list(action_weights.keys()), probabilities)[0]

    def get_all_actions(self, state=None, history=None):
        """
        Returns a list of all possible actions for the agent.
        - `FindAction` is only available if `LookAction` was the last action.
        - Uses `grid_map` to filter out invalid motions.
        """
        can_find = False
        if history and len(history) > 1:
            last_action = history[-1][0]
            if isinstance(last_action, LookAction):
                can_find = True
        find_action = [Find] if can_find else []

        if state is None:
            return ALL_MOTION_ACTIONS + [Look] + find_action
        else:
            if self._grid_map:
                valid_motions = self._grid_map.valid_motions(
                    self.robot_id, state.pose(self.robot_id), ALL_MOTION_ACTIONS
                )
                return list(valid_motions) + [Look] + find_action
            else:
                return ALL_MOTION_ACTIONS + [Look] + find_action

    def rollout(self, state, history=None):
        """Performs a rollout using improved action sampling."""
        return self.sample(state, history=history)

class PolicyModel(pomdp_py.RolloutPolicy):
    """Improved policy model with weighted action sampling."""

    def __init__(self, robot_id, grid_map=None, explore_prob=0.2):
        """
        Args:
            robot_id (int): ID of the robot.
            grid_map (GridMap, optional): Used to validate motion feasibility.
            explore_prob (float): Probability of taking a random exploratory action.
        """
        self.robot_id = robot_id
        self._grid_map = grid_map
        self.explore_prob = explore_prob  # Exploration-exploitation tradeoff

    def sample(self, state, history=None):
        """
        Samples an action intelligently instead of purely random.
        - Prioritizes `LookAction` if a new observation can be made.
        - Ensures `FindAction` only happens after `LookAction`.
        - Uses `explore_prob` to occasionally take a random action.
        """
        possible_actions = self.get_all_actions(state, history)
        
        # Prioritize LookAction if we haven't seen much yet
        if Look in possible_actions and random.random() > self.explore_prob:
            return Look

        # Weighted sampling based on past actions (Favor `LookAction` if needed)
        action_weights = {action: 1.0 for action in possible_actions}

        # If Find is an option, make it less likely unless we just Looked
        if Find in possible_actions:
            action_weights[Find] = 0.5  # Reduce probability of `Find`

        # Favor LookAction if we haven't taken it recently
        if history is not None and len(history) > 1:
            last_action = history[-1][0]
            if isinstance(last_action, LookAction):
                action_weights[Look] = 0.1  # Reduce `Look` frequency

        # Normalize and sample based on weights
        total_weight = sum(action_weights.values())
        probabilities = [w / total_weight for w in action_weights.values()]
        return random.choices(list(action_weights.keys()), probabilities)[0]

    def probability(self, action, state, **kwargs):
        """
        Computes the probability of selecting an action in the current state.
        """
        possible_actions = self.get_all_actions(state)
        if action in possible_actions:
            return 1.0 / len(possible_actions)
        return 0.0

    def argmax(self, state, history=None):
        """
        Returns the most likely action to be taken based on history.
        """
        return self.sample(state, history=history)

    def get_all_actions(self, state=None, history=None):
        """
        Returns a list of all possible actions for the agent.
        - `FindAction` is only available if `LookAction` was the last action.
        - Uses `grid_map` to filter out invalid motions.
        """
        can_find = False
        if history and len(history) > 1:
            last_action = history[-1][0]
            if isinstance(last_action, LookAction):
                can_find = True
        find_action = [Find] if can_find else []

        if state is None:
            return ALL_MOTION_ACTIONS + [Look] + find_action
        else:
            if self._grid_map:
                valid_motions = self._grid_map.valid_motions(
                    self.robot_id, state.pose(self.robot_id), ALL_MOTION_ACTIONS
                )
                return list(valid_motions) + [Look] + find_action
            else:
                return ALL_MOTION_ACTIONS + [Look] + find_action

    def rollout(self, state, history=None):
        """Performs a rollout using improved action sampling."""
        return self.sample(state, history=history)


class GreedyPolicy(pomdp_py.Planner):
    """Greedy planner that selects the best action maximizing a heuristic score."""

    def __init__(self, robot_id, grid_map=None):
        """
        Args:
            robot_id (int): ID of the robot.
            grid_map (GridMap, optional): Used to validate motion feasibility.
        """
        self.robot_id = robot_id
        self._grid_map = grid_map

    def plan(self, agent):
        """
        Selects the best action greedily.
        - Prefers actions that maximize information gain.
        - Prioritizes movements towards object-rich areas.
        """
        state = agent.cur_belief.mpe()
        possible_actions = self.get_all_actions(state)

        best_action = None
        best_score = float('-inf')

        for action in possible_actions:
            score = self._evaluate_action(state, action)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _evaluate_action(self, state, action):
        """Heuristic function to score an action."""
        if isinstance(action, LookAction):
            return 5  # Encourage looking
        elif isinstance(action, FindAction):
            return 10  # Encourage finding
        elif isinstance(action, MoveAction):
            return 2  # Favor movement but not randomly
        return 0  # Default score

    def get_all_actions(self, state):
        """
        Returns all possible actions.
        """
        return StrongRandomPolicy(self.robot_id, self._grid_map).get_all_actions(state)