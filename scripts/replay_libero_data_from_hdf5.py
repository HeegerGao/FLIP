''' replay demonstrations and save them as mp4 and state/actions.
'''
import sys
sys.path.append(".")
import argparse
import os
import h5py
import numpy as np
import json

from libero.utils.utils import update_env_kwargs, postprocess_model_xml
import cv2
import imageio
from libero.envs import *

current_file_path = os.path.abspath(__file__)
repo_dir = os.path.dirname(os.path.dirname(current_file_path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="data/libero_10", help="Directory containing the libero hdf5 files")
    parser.add_argument("--demo-suite", default="libero_10")
    parser.add_argument("--resolution", default=128)
    parser.add_argument("--use-camera-obs", default=True)
    parser.add_argument("--only-store-action", default=False, help="Only replay and store the actions and states")
    parser.add_argument("--libero_dir", default=f"{repo_dir}/libero")
    parser.add_argument("--use-depth", action="store_true")

    args = parser.parse_args()

    hdf5_files = []
    for dir_file in os.listdir(args.source_dir):
        if dir_file.endswith('.h5') or dir_file.endswith('.hdf5'):
            hdf5_files.append(os.path.join(args.source_dir, dir_file))

    for hdf5_path in hdf5_files:
        hdf5_name = os.path.basename(hdf5_path).split(".")[0]
        f = h5py.File(hdf5_path, "r")
        # data = f["data"]
        # print(data.attrs.keys())

        env_kwargs = json.loads(f["data"].attrs["env_args"])

        problem_info = json.loads(f["data"].attrs["problem_info"])
        problem_info["domain_name"]
        problem_name = problem_info["problem_name"]
        language_instruction = problem_info["language_instruction"]
        print(f"Language Instruction: {language_instruction}")

        # list of all demonstrations episodes
        demos = list(f["data"].keys())

        bddl_file_name = f["data"].attrs["bddl_file_name"][7:]

        output_parent_dir = os.path.join(args.source_dir, hdf5_name)
        os.makedirs(output_parent_dir, exist_ok=True)
        with open(os.path.join(output_parent_dir, "language_instruction.txt"), "w") as lf:
            lf.write(language_instruction)

        update_env_kwargs(
            env_kwargs["env_kwargs"],
            bddl_file_name=bddl_file_name,
            has_renderer=not args.use_camera_obs,
            has_offscreen_renderer=args.use_camera_obs,
            ignore_done=True,
            use_camera_obs=args.use_camera_obs,
            camera_depths=args.use_depth,
            camera_names=[
                "robot0_eye_in_hand",
                "agentview",
            ],
            reward_shaping=True,
            control_freq=20,
            camera_heights=args.resolution,
            camera_widths=args.resolution,
            camera_segmentations=None,
        )

        env = TASK_MAPPING[problem_name](
            **env_kwargs["env_kwargs"],
        )

        total_len = 0
        demos = demos

        print(f"Total number of episodes: {len(demos)}")

        for (i, ep) in enumerate(demos):
            print(f"Playing back {i}-th episode... (press ESC to quit)")
            # # select an episode randomly
            # read the model xml, using the metadata stored in the attribute for this episode
            model_xml = f["data/{}".format(ep)].attrs["model_file"]
            model_xml = model_xml.replace("/home/yifengz/workspace/libero-dev/chiliocosm", args.libero_dir)
            model_xml = model_xml.replace("/Users/bo/Desktop/libero-dev/chiliocosm", args.libero_dir)
            model_xml = model_xml.replace("/Users/yifengz/workspace/libero-dev/chiliocosm", args.libero_dir)
            reset_success = False
            while not reset_success:
                try:
                    env.reset()
                    reset_success = True
                except:
                    continue

            model_xml = postprocess_model_xml(model_xml, {})

            if not args.use_camera_obs:
                env.viewer.set_camera(0)

            # load the flattened mujoco states
            states = f["data/{}/states".format(ep)][()]
            actions = np.array(f["data/{}/actions".format(ep)][()])
            
            np.save(os.path.join(output_parent_dir, f"states_{ep}.npy"), states)
            np.save(os.path.join(output_parent_dir, f"actions_{ep}.npy"), actions)
            
            if args.only_store_action:
                continue
            num_actions = actions.shape[0]

            init_idx = 0
            env.reset_from_xml_string(model_xml)
            env.sim.reset()
            env.sim.set_state_from_flattened(states[init_idx])
            env.sim.forward()
            model_xml = env.sim.model.get_xml()

            agentview_images = []
            eye_in_hand_images = []

            agentview_depths = []
            eye_in_hand_depths = []

            valid_index = []

            for j, action in enumerate(actions):

                # obs, reward, done, info = env.step(action)
                env.sim.set_state_from_flattened(states[j])
                for _ in range(int(env.control_timestep / env.model_timestep)):
                    env.sim.forward()
                    # env.sim.step()
                    env._update_observables()
                if env.viewer is not None and env.renderer != "mujoco":
                    env.viewer.update()
                obs = env.viewer._get_observations() if env.viewer_get_obs else env._get_observations()

                if j < num_actions - 1:
                    # ensure that the actions deterministically lead to the same recorded states
                    state_playback = env.sim.get_state().flatten()
                    # assert(np.all(np.equal(states[j + 1], state_playback)))
                    err = np.linalg.norm(states[j] - state_playback)

                    if err > 0.01:
                        print(
                            f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}"
                        )

                valid_index.append(j)

                if args.use_camera_obs:
                    if args.use_depth:
                        agentview_depths.append(obs["agentview_depth"])
                        eye_in_hand_depths.append(obs["robot0_eye_in_hand_depth"])

                    agentview_images.append(obs["agentview_image"])
                    eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])
                else:
                    env.render()

            total_len += len(agentview_images)

            # save videos
            with imageio.get_writer(os.path.join(output_parent_dir, f"agentview_{ep}.mp4"), fps=24, codec='libx264') as writer:
                for idx, img in enumerate(agentview_images):
                    img = cv2.flip(img, 0)
                    writer.append_data(img)

            with imageio.get_writer(os.path.join(output_parent_dir, f"eye_in_hand_{ep}.mp4"), fps=24, codec='libx264') as writer:
                for idx, img in enumerate(eye_in_hand_images):
                    img = cv2.flip(img, 0)
                    writer.append_data(img)

            print(f"Episode {ep} has been saved to {output_parent_dir}")

        env.close()

        f.close()



if __name__ == "__main__":
    main()