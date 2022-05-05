import argparse
import os
import random
import gzip
import json

import numpy

import habitat
from habitat.core.env import Env
from habitat.datasets import make_dataset


class RandomAgent(habitat.Agent):
    def __init__(self, task_config: habitat.Config):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS

    def reset(self):
        pass

    def act(self, observations):
        return {"action": numpy.random.choice(self._POSSIBLE_ACTIONS)}


class HierarchicalAgent(habitat.Agent):
    # learns goals -> HLP -> LLA
    def __init__(self, task_config: habitat.Config):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.reset()

    def reset(self):
        self.current_goal = None
        self.current_plan = None

    def act(self, observations):
        return {"action": numpy.random.choice(self._POSSIBLE_ACTIONS)}


class ImitationAgent(habitat.Agent):
    # directly learns goals -> actions
    def __init__(self, task_config: habitat.Config):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.task_config_dataset = task_config.DATASET.DATA_PATH.format(split=task_config.DATASET.SPLIT)
        with gzip.open(self.task_config_dataset, "rt") as f:
            contents = json.loads(f.read())
        self.goal_obj_to_goal_obj_id = contents['category_to_task_category_id']
        self.goal_obj_id_to_goal_obj = {self.goal_obj_to_goal_obj_id[obj]: obj for obj in self.goal_obj_to_goal_obj_id}
        # # self.simulated_env = Env(config=task_config)
        # self._dataset = make_dataset(
        #     id_dataset=task_config.DATASET.TYPE, config=task_config.DATASET,
        # )
        # iter_option_dict = {
        #     k.lower(): v
        #     for k, v in task_config.ENVIRONMENT.ITERATOR_OPTIONS.items()
        # }
        # iter_option_dict["seed"] = task_config.SEED
        # breakpoint()
        # # TODO HOW to GET GOALS????
        # self._episode_iterator = self._dataset.get_episode_iterator(
        #     **iter_option_dict
        # )
        # self._current_episode = next(self._episode_iterator)
        # # for episode in self._dataset.episodes:
        # #     print(episode.goals[0].object_category)
        # # self.episode_idx = 0
        self.reset()

    def reset(self):
        self.goal_obj = None
        # self._current_episode = next(self._episode_iterator)
        # self.episode_idx += 1
        # self.simulated_env.reset()
        # self.current_goal_obj = None
        # for goal in self.simulated_env.current_episode.goals:
        #     if self.current_goal_obj is not None:
        #         assert self.current_goal_obj == goal.object_category
        #         self.current_goal_obj = goal.object_category

    def act(self, observations):
        breakpoint()
        self.goal_obj = self.goal_obj_id_to_goal_obj[observations['objectgoal'].item()]
        # self._env.current_episode.goals[0].object_category
        action = {"action": numpy.random.choice(self._POSSIBLE_ACTIONS)}
        # self.simulated_env.step(action)
        return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    args = parser.parse_args()

    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    agent = ImitationAgent(task_config=config)

    if args.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False)
    else:
        challenge = habitat.Challenge(eval_remote=True)

    challenge.submit(agent)


if __name__ == "__main__":
    main()
