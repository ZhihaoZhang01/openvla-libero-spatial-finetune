"""
Patched LIBERO eval client for openpi — aligned with OpenVLA eval protocol.

Changes vs upstream examples/libero/main.py:
  - num_tasks: limit to first N tasks (default None = all)
  - run_id_note / log_path: logging & video naming compatible with our sweep scripts
  - episode-indexed rollout MP4 filenames
"""

import collections
import dataclasses
import logging
import math
import pathlib
import re
from datetime import datetime
from typing import Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    num_tasks: Optional[int] = None  # None = all tasks; set 3 to match OpenVLA subset eval

    video_out_path: str = "data/libero/videos"
    log_path: str = ""  # if set, also write logs here (for results.csv parsing)
    run_id_note: str = "pi05-libero-eval"

    seed: int = 7


def _task_slug(description: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", description.strip().lower()).strip("_")
    return slug[:max_len] if slug else "task"


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    if args.log_path:
        pathlib.Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
        handlers = [logging.FileHandler(args.log_path, mode="w"), logging.StreamHandler()]
        logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    else:
        logging.basicConfig(level=logging.INFO, force=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    num_tasks_to_run = (
        num_tasks_in_suite if args.num_tasks is None else min(args.num_tasks, num_tasks_in_suite)
    )
    logging.info(f"Task suite: {args.task_suite_name} ({num_tasks_to_run}/{num_tasks_in_suite} tasks)")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    global_episode = 0

    for task_id in tqdm.tqdm(range(num_tasks_to_run)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            global_episode += 1
            logging.info(f"\nTask: {task_description}")

            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            done = False

            logging.info(f"Starting episode {episode_idx + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"replan_steps={args.replan_steps}, got {len(action_chunk)} actions"
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            success_flag = "True" if done else "False"
            mp4_name = (
                f"{run_ts}--episode={global_episode}--success={success_flag}"
                f"--task={_task_slug(task_description)}.mp4"
            )
            mp4_path = pathlib.Path(args.video_out_path) / mp4_name
            if replay_images:
                imageio.mimwrite(mp4_path, [np.asarray(x) for x in replay_images], fps=10)
                logging.info(f"Saved rollout MP4 at path {mp4_path}")

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            pct = total_successes / total_episodes * 100 if total_episodes else 0.0
            logging.info(f"# successes: {total_successes} ({pct:.1f}%)")

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    tyro.cli(eval_libero)
