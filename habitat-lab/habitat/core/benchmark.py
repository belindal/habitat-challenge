#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""Implements evaluation of ``habitat.Agent`` inside ``habitat.Env``.
``habitat.Benchmark`` creates a ``habitat.Env`` which is specified through
the ``config_env`` parameter in constructor. The evaluation is task agnostic
and is implemented through metrics defined for ``habitat.EmbodiedTask``.
"""

import os
from collections import defaultdict
from typing import Dict, Optional
import numpy as np

from habitat.config.default import get_config
from habitat.core.agent import Agent
from habitat.core.env import Env
from tqdm import tqdm
import json
import os
import time

import pandas as pd
category_mappings = pd.read_csv("/raid/lingo/bzl/habitat-challenge/habitat-sim/data/matterport_semantics/matterport_category_mappings.tsv", sep='    ', header=0)

# name_to_mpcat40_id = {category.category: category.mpcat40index for category in category_mappings}

# {'wall': 0, 'floor': 1, 'chair': 2, 'door': 3, 'table': 4, 'picture': 5, 'cabinet': 6, 'cushion': 7, 'window': 8, 'sofa': 9, 'bed': 10, 'curtain': 11, 'chest_of_drawers': 12, 'plant': 13, 'sink': 14, 'stairs': 15, 'ceiling': 16, 'toilet': 17, 'stool': 18, 'towel': 19, 'mirror': 20, 'tv_monitor': 21, 'shower': 22, 'column': 23, 'bathtub': 24, 'counter': 25, 'fireplace': 26, 'lighting': 27, 'beam': 28, 'railing': 29, 'shelving': 30, 'blinds': 31, 'gym_equipment': 32, 'seating': 33, 'board_panel': 34, 'furniture': 35, 'appliances': 36, 'clothes': 37, 'objects': 38, 'misc': 39, 'unlabeled': 40}

class Benchmark:
    r"""Benchmark for evaluating agents in environments."""

    def __init__(
        self, config_paths: Optional[str] = None, eval_remote: bool = False, overriden_args: list = None,
    ) -> None:
        r"""..

        :param config_paths: file to be used for creating the environment
        :param eval_remote: boolean indicating whether evaluation should be run remotely or locally
        """
        config_env = get_config(config_paths, overriden_args)
        self._eval_remote = eval_remote

        if self._eval_remote is True:
            self._env = None
        else:
            self._env = Env(config=config_env)

    def remote_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ):
        # The modules imported below are specific to habitat-challenge remote evaluation.
        # These modules are not part of the habitat-lab repository.
        import pickle
        import time

        import evalai_environment_habitat  # noqa: F401
        import evaluation_pb2
        import evaluation_pb2_grpc
        import grpc

        time.sleep(60)

        def pack_for_grpc(entity):
            return pickle.dumps(entity)

        def unpack_for_grpc(entity):
            return pickle.loads(entity)

        def remote_ep_over(stub):
            res_env = unpack_for_grpc(
                stub.episode_over(evaluation_pb2.Package()).SerializedEntity
            )
            return res_env["episode_over"]

        env_address_port = os.environ.get("EVALENV_ADDPORT", "localhost:8085")
        channel = grpc.insecure_channel(env_address_port)
        stub = evaluation_pb2_grpc.EnvironmentStub(channel)

        base_num_episodes = unpack_for_grpc(
            stub.num_episodes(evaluation_pb2.Package()).SerializedEntity
        )
        num_episodes = base_num_episodes["num_episodes"]

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0

        for _ in tqdm(range(num_episodes)):
            agent.reset()
            res_env = unpack_for_grpc(
                stub.reset(evaluation_pb2.Package()).SerializedEntity
            )

            while not remote_ep_over(stub):
                obs = res_env["observations"]
                action = agent.act(obs)

                res_env = unpack_for_grpc(
                    stub.act_on_environment(
                        evaluation_pb2.Package(
                            SerializedEntity=pack_for_grpc(action)
                        )
                    ).SerializedEntity
                )

            metrics = unpack_for_grpc(
                stub.get_metrics(
                    evaluation_pb2.Package(
                        SerializedEntity=pack_for_grpc(action)
                    )
                ).SerializedEntity
            )

            for m, v in metrics["metrics"].items():
                agg_metrics[m] += v
            count_episodes += 1

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        stub.evalai_update_submission(evaluation_pb2.Package())

        return avg_metrics

    def local_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0
        if agent.args.do_error_analysis:
            os.makedirs(agent.args.dump_location, exist_ok=True)
            results_file = os.path.join(agent.args.dump_location, "results.jsonl")
            with open(results_file, "a") as wf:
                wf.write("\n====\n")
        pbar = tqdm(range(num_episodes), desc="")
        total_timesteps = 0
        start_time = time.time()
        for i in pbar:
            observations = self._env.reset()
            sem_category_id_to_names = [obj.category.name() for obj_id, obj in enumerate(self._env.sim.semantic_scene.objects)]
            sem_category_id_to_mpcat40_ids = []
            for obj_id, obj in enumerate(self._env.sim.semantic_scene.objects):
                # cleanup
                obj_name = obj.category.name().lower().strip()
                obj_name = " ".join([term for term in obj_name.split(" ") if term != ''])
                if "window" in obj_name:
                    obj_name = "window"
                elif "towel" in obj_name:
                    obj_name = "towel"
                elif "door" in obj_name:
                    obj_name = "door"
                elif "bascet" in obj_name:
                    obj_name = "basket"
                elif "lamp" in obj_name:
                    obj_name = "lighting"
                elif "tv" in obj_name:
                    obj_name = "led tv"
                if sum((category_mappings["raw_category"] == obj_name) | (category_mappings["category"] == obj_name) | (category_mappings['mpcat40'] == obj_name)) == 0:
                    print(obj_name)
                    cat_id = 40  # unknown
                else:
                    cat_id = category_mappings['mpcat40index'][(category_mappings["raw_category"] == obj_name) | (category_mappings["category"] == obj_name) | (category_mappings['mpcat40'] == obj_name)].iloc[0] - 1
                sem_category_id_to_mpcat40_ids.append(cat_id)
            sem_category_id_to_mpcat40_ids = np.array(sem_category_id_to_mpcat40_ids)
            agent.reset()

            while not self._env.episode_over:
                if agent.args.do_error_analysis:
                    # Add these fields for error analysis
                    eps = self._env.current_episode
                    observations['origin'] = np.array(eps.start_position)
                    observations['rotation_world_start'] = np.array(eps.start_rotation)
                    observations['gt_goal_positions'] = [np.array(g.position) for g in eps.goals]
                    observations['success_distance'] = self._env.task.measurements.measures['success']._config.SUCCESS_DISTANCE
                    observations['self_position'] = self._env.task._sim.get_agent_state().position
                    observations['distance_to_goal'] = self._env.task.measurements.measures['distance_to_goal'].get_metric()
                    observations['env_id'] = '_'.join([eps.episode_id, os.path.split(eps.scene_id)[-1].split('.')[0], eps.goals[0].object_category])
                    observations['semantic_mapping'] = sem_category_id_to_mpcat40_ids
                action = agent.act(observations)
                total_timesteps += 1
                if agent.timestep % 100 == 0:
                    print(f"i={i}", f"timestep={agent.timestep}", f"avg time/step = {(time.time() - start_time) / total_timesteps}")
                observations = self._env.step(action)
                # if self._env.task.measurements.measures['distance_to_goal']._metric:
                # false negative (but what if taret object not yet in sight????)
                # self._env.task._sim.get_agent_state().position

            metrics = self._env.get_metrics()
            for m, v in metrics.items():
                if isinstance(v, dict):
                    for sub_m, sub_v in v.items():
                        agg_metrics[m + "/" + str(sub_m)] += sub_v
                else:
                    agg_metrics[m] += v
            if agent.args.do_error_analysis:
                result = {'env_id': '_'.join([eps.episode_id, os.path.split(eps.scene_id)[-1].split('.')[0], eps.goals[0].object_category]), 'metrics': metrics, 'target': action['objectgoal']}
                for k in action:
                    if k not in ['objectgoal', 'action', 'success']:
                        result[k] = action[k]
                # 'saw_target_frames': action['saw_target'], 'nearby_objs': action['nearby_objs'], 
                # write to file
                with open(results_file, "a") as wf:
                    wf.write(json.dumps(result)+"\n")
            count_episodes += 1
            pbar.set_description(' '.join([
                f'{m}={agg_metrics[m] / count_episodes:.2f}'
                if not isinstance(agg_metrics[m], dict)
                else f'{m}={json.dumps({sub_m: agg_metrics[m][sub_m] / count_episodes for sub_m in agg_metrics[m]})}'
                for m in agg_metrics
            ] + [f"time/step={round((time.time() - start_time) / total_timesteps, 2)}"]))

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
        if agent.args.do_error_analysis:
            # log metrics
            with open(agent.args.do_error_analysis, "a") as wf:
                wf.write("Metrics: "+json.dumps(avg_metrics)+"\n")

        return avg_metrics

    def evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        if self._eval_remote is True:
            return self.remote_evaluate(agent, num_episodes)
        else:
            return self.local_evaluate(agent, num_episodes)
