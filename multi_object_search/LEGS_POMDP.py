import pomdp_py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from multi_object_search.env.env import *
from multi_object_search.env.visual import *
from multi_object_search.agent.agent import *
from multi_object_search.agent.belief import belief_update
from multi_object_search.example_worlds import *
from multi_object_search.domain.observation import *
from multi_object_search.models.components.grid_map import *

import argparse
import time
import random


class MosOOPOMDP(pomdp_py.OOPOMDP):
    """
    A MosOOPOMDP is instantiated given a string description
    of the search world, sensor descriptions for robots,
    and the necessary parameters for the agent's models.
    """

    def __init__(
        self,
        robot_id,
        env=None,
        grid_map=None,
        sensors=None,
        sigma=0.01,
        epsilon=1,
        belief_rep="histogram",
        prior={},
        num_particles=100,
        agent_has_map=False,
    ):
        """
        Args:
            robot_id (int or str): The agent solving the POMDP.
            env (MosEnvironment): The environment. If `None`, we construct it from `grid_map`.
            grid_map (str): Search space description.
            sensors (dict): Map from robot character to sensor string.
            sigma, epsilon: Observation model parameters.
            belief_rep (str): "histogram" or "particles".
            prior (dict or str): Dictionary defining prior belief or string ("uniform", "informed").
            num_particles (int): Used for particle-based belief representation.
            agent_has_map (bool): If `True`, the agent knows the occupancy grid map.
        """
        if env is None:
            assert grid_map is not None and sensors is not None, (
                "Since env is not provided, you must provide grid_map and sensors."
            )
            worldstr = equip_sensors(grid_map, sensors)
            dim, robots, objects, obstacles, sensors = interpret(worldstr)
            init_state_2d = {}
            for objid, objstate in objects.items():
                x, y = objstate.pose  # Extract only x, y
                init_state_2d[objid] = ObjectState(objid, objstate.objclass, (x, y))

            for robotid, robstate in robots.items():
                x, y, th = robstate.pose
                init_state_2d[robotid] = RobotState(robotid, (x, y, th), robstate.objects_found, None)

            init_state = MosOOState(init_state_2d)
            env = MosEnvironment(dim, init_state, sensors, obstacles=obstacles)
            robot_id = interpret_robot_id(robot_id)
            if None in [dim, robots, objects, obstacles, sensors]:
                raise ValueError(f"Environment parsing failed: {dim}, {robots}, {objects}, {obstacles}, {sensors}")
            print(f"Robots: {robots}, Robot ID: {robot_id}")
            print(f"Sensors: {sensors}")
        # Ensure sensor exists for the agent
        if robot_id not in env.sensors:
            raise ValueError(f"Sensor for robot_id {robot_id} is missing in environment!")

        robot_sensor = env.sensors[robot_id]  # Extract sensor

        # Construct prior belief
        if isinstance(prior, str):
            if prior == "uniform":
                prior = {}
            elif prior == "informed":
                prior = {
                    objid: {env.state.pose(objid): 1.0}
                    for objid in env.target_objects
                }

        # Initialize the agent
        robot_id = robot_id if isinstance(robot_id, int) else interpret_robot_id(robot_id)
        grid_map = (
            GridMap(env.width, env.length, {objid: env.state.pose(objid) for objid in env.obstacles})
            if agent_has_map else None
        )
        agent = MosAgent(
            robot_id,
            env.state.object_states[robot_id],
            env.target_objects,
            (env.width, env.length),
            robot_sensor,  # 🔥 Ensure a valid sensor is passed!
            sigma=sigma,
            epsilon=epsilon,
            belief_rep=belief_rep,
            prior=prior,
            num_particles=num_particles,
            grid_map=grid_map,
        )
        super().__init__(agent, env, name=f"MOS({env.width},{env.length},{len(env.target_objects)})")


### **Solve Function**
def solve(
    problem, max_depth=10, discount_factor=0.99, planning_time=1.0,
    exploration_const=1000, visualize=True, max_time=500, max_steps=500
):
    """
    Solves the POMDP using POUCT/POMCP planning.
    """
    random_objid = random.choice(list(problem.env.target_objects))
    random_object_belief = problem.agent.belief.object_beliefs[random_objid]

    planner = pomdp_py.POUCT(
        max_depth=max_depth,
        discount_factor=discount_factor,
        planning_time=planning_time,
        exploration_const=exploration_const,
        rollout_policy=problem.agent.policy_model,
    ) if isinstance(random_object_belief, pomdp_py.Histogram) else pomdp_py.POMCP(
        max_depth=max_depth,
        discount_factor=discount_factor,
        planning_time=planning_time,
        exploration_const=exploration_const,
        rollout_policy=problem.agent.policy_model,
    )

    robot_id = problem.agent.robot_id
    if visualize:
        viz = MosViz(problem.env, controllable=False)
        if not viz.on_init():
            raise Exception("Environment failed to initialize")
        viz.update(robot_id, None, None, None, problem.agent.cur_belief)
        viz.on_render()

    _time_used = 0
    _find_actions_count = 0
    _total_reward = 0

    for i in range(max_steps):
        _start = time.time()
        real_action = planner.plan(problem.agent)
        _time_used += time.time() - _start
        if _time_used > max_time:
            break

        reward = problem.env.state_transition(real_action, execute=True, robot_id=robot_id)

        _start = time.time()
        real_observation = problem.env.provide_observation(problem.agent.observation_model, real_action)

        problem.agent.clear_history()
        problem.agent.update_history(real_action, real_observation)
        belief_update(problem.agent, real_action, real_observation, problem.env.state.object_states[robot_id], planner)
        _time_used += time.time() - _start

        _total_reward += reward
        if isinstance(real_action, FindAction):
            _find_actions_count += 1

        print(f"==== Step {i+1} ====")
        print(f"Action: {real_action}")
        print(f"Observation: {real_observation}")
        print(f"Reward: {reward}")
        print(f"Total Reward: {_total_reward}")
        print(f"Find Actions Count: {_find_actions_count}")

        if visualize:
            viz.update(robot_id, real_action, real_observation, None, problem.agent.cur_belief)
            viz.on_loop()
            viz.on_render()

        if set(problem.env.state.object_states[robot_id].objects_found) == problem.env.target_objects:
            print("Done!")
            break
        if _find_actions_count >= len(problem.env.target_objects):
            print("FindAction limit reached.")
            break
        if _time_used > max_time:
            print("Maximum time reached.")
            break


if __name__ == "__main__":
    grid_map, robot_char = random_world(10, 10, 5, 2)
    print("Grid map World:\n", grid_map)
    problem = MosOOPOMDP( 'r', grid_map=grid_map, sensors={robot_char: "proximity"})
    solve(problem, visualize=True)