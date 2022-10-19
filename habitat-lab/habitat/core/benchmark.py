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
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.tasks.nav.object_nav_task import ObjectGoal

import pandas as pd
import cv2
from PIL import Image
from quaternion import quaternion
import csv
category_mappings = pd.read_csv("/raid/lingo/bzl/habitat-challenge/habitat-sim/data/matterport_semantics/matterport_category_mappings.tsv", sep='    ', header=0)

possible_labels = ["OK", "Not OK"]
# (n_items, n_rooms)
hm3d_items2room_cooccurs = np.load("/raid/lingo/bzl/Stubborn/rednet-finetuning/figures/gpt3_room_obj_plausimplaus_hm3d_cooccurence.npy", allow_pickle=True)
hm3d_items = [
    'wall', 'floor', 'chair', 'door', 'table', # 5
    'picture', 'cabinet', 'cushion', 'window', 'sofa',
    'bed', 'curtain', 'chest_of_drawers', 'plant', 'sink',
    'stairs', 'ceiling', 'toilet', 'stool', 'towel', # 20
    'mirror', 'tv_monitor', 'shower', 'column', 'bathtub',
    'counter', 'fireplace', 'lighting', 'beam', 'railing',
    'shelving', 'blinds', 'gym_equipment', # 33
    'seating', 'board_panel',
    'furniture', 'appliances', 'clothes', 'objects', 'misc',
    'unlabeled' # 41
]
rooms = [
     'bedroom', 'bathroom', 'closet', 'hallway', 'living room', 'dining room',
     'kitchen', 'entrance', 'basement', 'staircase', 'laundry room',
     'office', 'patio', 'stairs', 'lounge', 'sauna', 'shower room',
     'playroom', 'garage', 'shed', 'storage', 'baby room', 'study',
     "kid's room", 'gym', 'shower', 'attic', 'dressing room', 'pantry']
gt_hm3d_items2room_cooccurs = np.load("/raid/lingo/bzl/Stubborn/rednet-finetuning/figures/real_room_obj_plausimplaus_hm3d_cooccurence.npy", allow_pickle=True)

# name_to_mpcat40_id = {category.category: category.mpcat40index for category in category_mappings}

# {'wall': 0, 'floor': 1, 'chair': 2, 'door': 3, 'table': 4, 'picture': 5, 'cabinet': 6, 'cushion': 7, 'window': 8, 'sofa': 9, 'bed': 10, 'curtain': 11, 'chest_of_drawers': 12, 'plant': 13, 'sink': 14, 'stairs': 15, 'ceiling': 16, 'toilet': 17, 'stool': 18, 'towel': 19, 'mirror': 20, 'tv_monitor': 21, 'shower': 22, 'column': 23, 'bathtub': 24, 'counter': 25, 'fireplace': 26, 'lighting': 27, 'beam': 28, 'railing': 29, 'shelving': 30, 'blinds': 31, 'gym_equipment': 32, 'seating': 33, 'board_panel': 34, 'furniture': 35, 'appliances': 36, 'clothes': 37, 'objects': 38, 'misc': 39, 'unlabeled': 40}
def convert_to_gps_coords(
    position, episode_start_position, episode_start_rotation
):
    origin = np.array(episode_start_position, dtype=np.float32)
    rotation_world_start = quaternion_from_coeff(episode_start_rotation)

    position = quaternion_rotate_vector(
        rotation_world_start.inverse(), position - origin
    )
    return np.array(
        [-position[2], position[0]], dtype=np.float32
    )
    # if self._dimensionality == 2:
    # else:
    #     return position.astype(np.float32)

def convert_to_world_coords(
    position, episode_start_position, episode_start_rotation
):
    origin = np.array(episode_start_position, dtype=np.float32)
    rotation_world_start = quaternion_from_coeff(episode_start_rotation)
    position = np.array([position[1], 0.0, -position[0]], dtype=np.float32)

    position = quaternion_rotate_vector(
        rotation_world_start, position
    ) + origin
    return np.array(position, dtype=np.float32)


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
        extra_objs_success = []

        count_episodes = 0
        existing_results = set()
        if agent.args.do_error_analysis:
            os.makedirs(agent.args.dump_location, exist_ok=True)
            results_file = os.path.join(agent.args.dump_location, "results.jsonl")
            if os.path.exists(results_file):
                with open(results_file) as f:
                    for line in f:
                        if "====" in line or line.strip() == "": continue
                        line = json.loads(line)
                        existing_results.add(line['env_id'])
            with open(results_file, "a") as wf:
                wf.write("\n====\n")
        pbar = tqdm(range(num_episodes), desc="")
        total_timesteps = 0
        start_time = time.time()
        seen_scenes = set()
        n_nonskipped = 0
        room_annotations = csv.reader(open("room_classification_images/room_annotations.csv"))
        room2roomtype = {}
        roomtype2room = {}
        for annotation in room_annotations:
            if len(annotation[2].strip().strip('-')) == 0 or "unclear" in annotation[2]: continue
            room_types = annotation[2].split(',')
            if annotation[0] not in room2roomtype:
                room2roomtype[annotation[0]] = {}
            if annotation[1] not in room2roomtype[annotation[0]]:
                room2roomtype[annotation[0]][annotation[1]] = []
            for room_type in room_types:
                room_type = room_type.strip()
                room2roomtype[annotation[0]][annotation[1]].append(room_type)
                if room_type not in roomtype2room:
                    roomtype2room[room_type] = {}
                if annotation[0] not in roomtype2room[room_type]:
                    roomtype2room[room_type][annotation[0]] = {}
                roomtype2room[room_type][annotation[0]][annotation[1]] = 1.0 / len(room_types)
        room_found = []
        room_accuracy = []
        for i in pbar:
            observations = self._env.reset()
            eps = self._env.current_episode

            # skip ones that aren't on the same floor as start
            gt_goal_locations = [
                np.array(goal.position) for goal in eps.goals if abs(goal.position[1] - eps.start_position[1]) < 0.5
            ]
            if len(gt_goal_locations) == 0:
                continue
            goal_obj = eps.goals[0].object_category
            goal_ids = [goal.object_id for goal in eps.goals]
            env_id = '_'.join([eps.episode_id, os.path.split(eps.scene_id)[-1].split('.')[0], goal_obj])
            # if os.path.split(eps.scene_id)[-1].split('.')[0] != "mL8ThkuaVTM": continue
            # if os.path.split(eps.scene_id)[-1].split('.')[0] != "Nfvxx8J5NCo": continue
            # if os.path.split(eps.scene_id)[-1].split('.')[0] != "5cdEh9F2hJL": continue
            # if os.path.split(eps.scene_id)[-1].split('.')[0] != "q3zU7Yy5E5s": continue
            # if env_id != "32_wcojb4TFT35_sofa": continue
            if env_id in existing_results: continue
            # if eps.scene_id in seen_scenes: continue
            sem_category_id_to_names = [obj.category.name() for obj_id, obj in enumerate(self._env.sim.semantic_scene.objects)]
            sem_category_id_to_mpcat40_ids = []

            all_goal_objs = []
            extra_goal_objs = []
            missing_goal = False
            for obj_id, obj in enumerate(self._env.sim.semantic_scene.objects):
                # cleanup
                obj_name = obj.category.name().lower().strip()
                obj_name = " ".join([term for term in obj_name.split(" ") if term != ''])
                if "window" in obj_name: obj_name = "window"
                elif "towel" in obj_name: obj_name = "towel"
                elif "door" in obj_name: obj_name = "door"
                elif "bascet" in obj_name: obj_name = "basket"
                elif "lamp" in obj_name: obj_name = "lighting"
                elif "tv" in obj_name: obj_name = "tv"
                elif "device" in obj_name: obj_name = "misc"
                elif obj_name == "lmap": obj_name = "lighting"
                elif obj_name == "dorr": obj_name = "door"
                elif obj_name == "unknwn": obj_name = "unknown"
                elif obj_name == "couch": obj_name = "sofa"
                elif "sofa" in obj_name: obj_name = "sofa"
                elif "plant" in obj_name: obj_name = "plant"
                elif "chair" in obj_name: obj_name = "chair"
                if sum((category_mappings["raw_category"] == obj_name) | (category_mappings["category"] == obj_name) | (category_mappings['mpcat40'] == obj_name)) == 0:
                    # print(obj_name)
                    cat_id = 40  # unknown
                else:
                    cat_id = category_mappings['mpcat40index'][(category_mappings["raw_category"] == obj_name) | (category_mappings["category"] == obj_name) | (category_mappings['mpcat40'] == obj_name)].iloc[0] - 1
                sem_category_id_to_mpcat40_ids.append(cat_id)
                if obj_name == goal_obj:
                    all_goal_objs.append(obj)
                    if obj_id not in goal_ids:
                        extra_goal_objs.append(obj)
            # # eps.goals.extend([ObjectGoal(object_id=obj.id.split('_')[-1], object_name=obj.category.name(), object_category=obj.id, position=obj.aabb.center) for obj in all_goal_objs])
            # if len(all_goal_objs) != len(eps.goals) or missing_goal:
            #     # # discongruity in annotation (potentially wrong annotation), skip
            #     # with open ("bad_episodes.txt", "a") as wf:
            #     #     wf.write(env_id + "\n")
            #     continue
            sem_category_id_to_mpcat40_ids = np.array(sem_category_id_to_mpcat40_ids)
            room_id_to_info = {}
            rooms_containing_goal = []
            rooms_on_floor = []
            for room_id, room in enumerate(self._env.sim.semantic_scene.regions):
                if int(room.id.split("_")[-1]) == -1: continue
                obj_centers = np.stack([obj.aabb.center for obj in room.objects])
                floors = [obj for obj in room.objects if obj.category.name() == "floor"]
                ceilings = [obj for obj in room.objects if obj.category.name() == "ceiling"]
                if len(floors) > 0 and len(ceilings) > 0:
                    floor_centers = np.stack([floor.aabb.center for floor in floors])
                    floor_sizes = np.stack([floor.aabb.sizes for floor in floors])
                    floor_max_xyz = (floor_centers + floor_sizes / 2)
                    floor_min_xyz = (floor_centers - floor_sizes / 2)
                    floor_xyz_ranges = np.stack([floor_min_xyz, floor_max_xyz], axis=-1)
                    ceiling_centers = np.stack([ceil.aabb.center for ceil in ceilings])
                    ceiling_sizes = np.stack([ceil.aabb.sizes for ceil in ceilings])
                    ceiling_max_xyz = (ceiling_centers + ceiling_sizes / 2)
                    ceiling_min_xyz = (ceiling_centers - ceiling_sizes / 2)
                    ceiling_xyz_ranges = np.stack([ceiling_min_xyz, ceiling_max_xyz], axis=-1)
                    room_bbs = floor_xyz_ranges
                    room_bbs[:,1,1] = ceiling_xyz_ranges[:,1,1].max()
                    room_bb = room_bbs[0]
                    room_bb[:,0] = room_bbs[:,:,0].min(0)
                    room_bb[:,1] = room_bbs[:,:,1].max(0)
                else:
                    # print(eps.scene_id, room.id)
                    obj_centers = np.stack([obj.aabb.center for obj in room.objects])
                    objs_max_xyz = obj_centers.max(0)
                    objs_min_xyz = obj_centers.min(0)
                    room_bb = np.stack([objs_min_xyz, objs_max_xyz], axis=-1)
                room_id_to_info[room.id] = {'bb': room_bb, 'name': room2roomtype[eps.scene_id].get(room.id, ["Unknown"])}
                """
                get rooms with goal & rooms on floor
                """
                for goal_location in gt_goal_locations:
                    if (goal_location >= room_bb[:,0]).all() and (goal_location <= room_bb[:,1]).all():
                        rooms_containing_goal.append(room.id)
                if eps.start_position[1] >= room_bb[1,0] and eps.start_position[1] <= room_bb[1,0] + 1.0:  #room_bb[1,1]:
                    rooms_on_floor.append(room.id)
                """ # generate snapshots of rooms
                room_center = room.aabb.center
                agent_rotation = self._env.task._sim.get_agent_state().rotation
                room_corner_views = [
                    ["corner", room_bb[:,0], quaternion(-0.3826834, 0, 0.9238795, 0)],
                    ["corner", np.array([room_bb[0,1], room_bb[1,0], room_bb[2,1]]), quaternion(0.9238795, 0, 0.3826834, 0)],
                    ["corner", np.array([room_bb[0,1], room_bb[1,0], room_bb[2,0]]), quaternion(0.3826834, 0, 0.9238795, 0)],
                    ["corner", np.array([room_bb[0,0], room_bb[1,0], room_bb[2,1]]), quaternion(-0.9238795, 0, 0.3826834, 0)],
                ]
                room_center = room_bb.mean(-1)
                room_center[1] = room_bb[1,0]
                snapshots = [
                    ["center", room_center, quaternion(0.9238795, 0, 0.3826834, 0)], ["center",room_center, quaternion(-0.9238795, 0, 0.3826834, 0)],
                    ["center", room_center, quaternion(0.7071068, 0, 0.7071068, 0)], ["center",room_center, quaternion(-0.7071068, 0, 0.7071068, 0)],
                    ["center", room_center, quaternion(0.3826834, 0, 0.9238795, 0)], ["center",room_center, quaternion(-0.3826834, 0, 0.9238795, 0)],
                    ["center", room_center, quaternion(1.0, 0, 0, 0)], ["center",room_center, quaternion(-1.0, 0, 0, 0).inverse()],
                    *room_corner_views
                ]
                directory = os.path.join("room_classification_images", f"{os.path.split(os.path.split(eps.scene_id)[0])[-1]}", f"room{room.id}")
                os.makedirs(directory, exist_ok=True)
                for position_name, position, rotation in snapshots:
                    observations_in_room = self._env.sim.get_observations_at(position=position, rotation=rotation)
                    rgb_in_room = Image.fromarray(observations_in_room['rgb'])
                    rgb_in_room.save(os.path.join(directory, f"{position_name}_rot{rotation.tolist()}.png"))
                objects_in_room = [obj.category.name() for obj in room.objects]
                # for obj in room.objects: print(obj.category.name())
                with open("room_classification_images/saved_val_annotations.txt", "a") as wf:
                    wf.write(json.dumps({
                        "scene_id": eps.scene_id, "room_id": room.id, "label": "",
                        "images": directory, "center_XYZ": room_center.tolist(), "aabb_XYZ": room_bb.tolist(),
                        "scene_url": f"https://aihabitat.org/datasets/hm3d/{os.path.split(os.path.split(eps.scene_id)[0])[-1]}/index.html",
                        "objects": objects_in_room,
                    })+"\n")
                    wf.flush()
                # """
            # # # TODO DELETE
            # seen_scenes.add(eps.scene_id)
            # continue
            if agent.args.explore_room_order == "lm_prior" or agent.args.explore_room_order == "gt_prior":
                prior_rooms_containing_goal = []
                if agent.args.explore_room_order == "lm_prior":
                    room2score = {rooms[i]: hm3d_items2room_cooccurs[hm3d_items.index(goal_obj),i] for i in range(len(rooms))}
                    threshold = 0.3
                    sorted_room_types = sorted(rooms, key=room2score.get, reverse=True)
                elif agent.args.explore_room_order == "gt_prior":
                    room2score = {rooms[i]: gt_hm3d_items2room_cooccurs[hm3d_items.index(goal_obj),i] for i in range(len(rooms))}
                    threshold = 0.0
                    sorted_room_types = sorted(rooms, key=room2score.get, reverse=True)
                """
                find best room -- top of sorted list (if not > 0.95, delete)
                """
                for room_type in sorted_room_types:
                    rooms_of_type = roomtype2room.get(room_type, {}).get(eps.scene_id, {})
                    accessible_rooms_of_type = list(set(rooms_on_floor).intersection(set(rooms_of_type.keys())))
                    if len(accessible_rooms_of_type) > 0 or room2score[room_type] < threshold: break
                if room2score[room_type] >= threshold:
                    # sort by "proportion" of region dedicated to this room
                    accessible_rooms_of_type = sorted(accessible_rooms_of_type, key=rooms_of_type.get, reverse=True)
                    prior_rooms_containing_goal.extend(accessible_rooms_of_type)
                    # if goal_obj == "chair": breakpoint()
                    # # is confidence we'll find obj in this room type > 0.95?
                    # room_goal_cooccur_confidences = max(room2score[room_type] for room_type in room2roomtype[eps.scene_id][lmprior_rooms_containing_goal[0]])
                    # if room_goal_cooccur_confidences < 0.95:
                    #     lmprior_rooms_containing_goal = []
                room_intersection = set(prior_rooms_containing_goal).intersection(set(rooms_containing_goal))
                room_found.append(len(prior_rooms_containing_goal) > 0)
                room_accuracy.append(len(room_intersection) > 0)
                """
                with open(f"{agent.args.dump_location}/objroom_cooccurs_nav.jsonl", "a") as wf:
                    real_roomtypes = [room2roomtype[eps.scene_id].get(room, "Unknown") for room in rooms_containing_goal]
                    pred_roomtypes = [room2roomtype[eps.scene_id].get(room, "Unknown") for room in prior_rooms_containing_goal]
                    accessible_rooms = [room2roomtype[eps.scene_id].get(room, "Unknown") for room in rooms_on_floor]
                    wf.write(json.dumps({"goal": goal_obj, "actual rooms": real_roomtypes, "pred room": pred_roomtypes, "accessible rooms": accessible_rooms, "eps": env_id})+"\n")
                    # breakpoint()
                pbar.set_description(f"Found: {sum(room_found) / len(room_found)}; acc: {sum(room_accuracy) / len(room_accuracy)}")
                continue
                # """

            agent.reset()

            while not self._env.episode_over:
                observations['semantic_mapping'] = sem_category_id_to_mpcat40_ids
                # TODO change to gps locations
                observations['room_id_to_info'] = room_id_to_info
                observations['distance_to_goal'] = self._env.task.measurements.measures['distance_to_goal'].get_metric()
                observations['gt_goal_positions'] = gt_goal_locations  #[np.array(g.position) for g in eps.goals]
                if agent.args.explore_room_order == "lm_prior" or agent.args.explore_room_order == "gt_prior":
                    observations['room2score'] = room2score
                if agent.args.explore_room_order == "gt":
                    observations['goal_rooms'] = rooms_containing_goal
                    # sort goal rooms by distance to current gps position
                    observations['goal_rooms'].sort(key=lambda room: ((observations["gps"] - convert_to_gps_coords(room_id_to_info[room]['bb'].mean(-1), eps.start_position, eps.start_rotation))**2).sum())
                elif agent.args.explore_room_order == "distance":
                    observations['goal_rooms'] = rooms_on_floor
                    # sort goal rooms by distance to current gps position
                    observations['goal_rooms'].sort(key=lambda room: ((observations["gps"] - convert_to_gps_coords(room_id_to_info[room]['bb'].mean(-1), eps.start_position, eps.start_rotation))**2).sum())
                elif agent.args.explore_room_order in ["lm_prior", "gt_prior"]:
                    observations['goal_rooms'] = prior_rooms_containing_goal
                    # sort goal rooms by distance to current gps position
                    observations['goal_rooms'].sort(key=lambda room: (rooms_of_type.get, ((observations["gps"] - convert_to_gps_coords(room_id_to_info[room]['bb'].mean(-1), eps.start_position, eps.start_rotation))**2).sum()))
                # breakpoint()
                """
                # add in rest of rooms (also sorted by distance)
                rooms_on_floor.sort(key=lambda room: ((observations["gps"] - convert_to_gps_coords(room_id_to_info[room].mean(-1), eps.start_position, eps.start_rotation))**2).sum())
                for room in rooms_on_floor:
                    if room not in observations['goal_rooms']:
                        observations['goal_rooms'].append(room)
                # """
                observations['accessible_rooms'] = rooms_on_floor
                observations['gt_goal_name'] = goal_obj  #[np.array(g.position) for g in eps.goals]
                observations['start'] = {'position': np.array(eps.start_position), 'rotation': np.array(eps.start_rotation)}
                observations['self_abs_position'] = self._env.task._sim.get_agent_state().position
                observations['env_id'] = env_id
                # if agent.args.do_error_analysis:
                #     # Add these fields for error analysis
                #     observations['success_distance'] = self._env.task.measurements.measures['success']._config.SUCCESS_DISTANCE
                #     observations['env_id'] = env_id
                try:
                    assert (observations['gps'] == convert_to_gps_coords(self._env.task._sim.get_agent_state().position, eps.start_position, eps.start_rotation)).all()
                except:
                    breakpoint()
                # transform room to gps coordinates?
                action = agent.act(observations)
                total_timesteps += 1
                if agent.timestep % 100 == 0:
                    print(f"i={i}", f"env={env_id}", f"timestep={agent.timestep}", f"time/step = {(time.time() - start_time) / total_timesteps}")
                observations = self._env.step(action)
                # if self._env.task.measurements.measures['distance_to_goal']._metric:
                # false negative (but what if taret object not yet in sight????)
                # self._env.task._sim.get_agent_state().position

            existing_results.add(env_id)
            metrics = self._env.get_metrics()
            extra_goals_success = False
            if len(extra_goal_objs) > 0:
                extra_goals_success = max([
                    (np.absolute((self._env.task._sim.get_agent_state().position - goal_obj.aabb.center)) < (goal_obj.aabb.sizes / 2) + 0.1).all()
                for goal_obj in extra_goal_objs]) and action['stop_reason'] == "found goal"
            metrics["all_goal_successes"] = (len(extra_goal_objs) > 0 and extra_goals_success) or agg_metrics["success"]
            for m, v in metrics.items():
                if isinstance(v, dict):
                    for sub_m, sub_v in v.items():
                        agg_metrics[m + "/" + str(sub_m)] += sub_v
                else:
                    agg_metrics[m] += v
            if agent.args.do_error_analysis:
                result = {
                    'env_id': env_id, 'metrics': metrics, 'target': goal_obj,
                    "final_position": self._env.task._sim.get_agent_state().position.tolist(),
                    "extra_goal_positions": [{'center': ego.aabb.center.tolist(), 'size': ego.aabb.sizes.tolist()} for ego in extra_goal_objs],
                }
                for k in action:
                    if k not in ['objectgoal', 'action', 'success']:
                        result[k] = action[k]
                # 'saw_target_frames': action['saw_target'], 'nearby_objs': action['nearby_objs'], 
                # write to file
                if agent.args.explore_room_order == "lm_prior" or agent.args.explore_room_order == "gt_prior":
                    real_roomtypes = [room2roomtype[eps.scene_id].get(room, "Unknown") for room in rooms_containing_goal]
                    pred_roomtypes = [room2roomtype[eps.scene_id].get(room, "Unknown") for room in prior_rooms_containing_goal]
                    accessible_rooms = [room2roomtype[eps.scene_id].get(room, "Unknown") for room in rooms_on_floor]
                    result = {**result, "actual rooms": real_roomtypes, "pred room": pred_roomtypes, "accessible rooms": accessible_rooms, "correctly pred room": list(room_intersection)}
                with open(results_file, "a") as wf:
                    wf.write(json.dumps(result)+"\n")
            count_episodes += 1
            pbar.set_description(' '.join([
                f'{m}={agg_metrics[m] / count_episodes:.2f}'
                if not isinstance(agg_metrics[m], dict)
                else f'{m}={json.dumps({sub_m: agg_metrics[m][sub_m] / count_episodes for sub_m in agg_metrics[m]})}'
                for m in ["success", "all_goal_successes", "distance_to_goal"]
            ] + [f"room acc={sum(room_accuracy) / len(room_accuracy) if len(room_accuracy) > 0 else 0:.2f}"] + [f"time/step={round((time.time() - start_time) / total_timesteps, 2)}"]))

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
        # if agent.args.do_error_analysis:
        #     # log metrics
        #     with open(agent.args.do_error_analysis, "a") as wf:
        #         wf.write("Metrics: "+json.dumps(avg_metrics)+"\n")

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
