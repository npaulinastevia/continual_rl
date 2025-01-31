# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Taken from https://github.com/facebookresearch/torchbeast/blob/3f3029cf3d6d488b8b8f952964795f451a49048f/torchbeast/core/environment.py
# and modified slightly
"""The environment class for MonoBeast."""

import torch
import numpy as np


def _format_frame(frame):
    frame = frame.to_tensor()  # Convert from LazyFrames
    return frame.view((1, 1) + frame.shape)  # (...) -> (T,B,...).


class Environment:
    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None


    def initial(self):
        initial_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.

        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.zeros(1, 1, dtype=torch.uint8)  # Originally this was ones, which makes there be 0 reward episodes
        if len(self.gym_env.reset().shape)==1:
            initial_frame=torch.from_numpy(self.gym_env.reset())
        elif len(self.gym_env.reset().shape)==2:
            initial_frame=torch.from_numpy(self.gym_env.reset())
            #initial_frame = torch.flatten(initial_frame)[:201]
            #initial_frame=initial_frame[0:7,0:7]
            initial_frame= initial_frame.unsqueeze(0)
            initial_frame = initial_frame.unsqueeze(0)

   
        else:
            initial_frame = _format_frame(self.gym_env.reset())

        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,

        )

    def step(self, action,return_rr=False):

        if return_rr:
            frame, reward, done, prior_info,rr, map = self.gym_env.step(action.item(), return_rr=return_rr)
            pik=self.gym_env.picked
            match=self.gym_env.match_id
        else:
            frame, reward, done, prior_info = self.gym_env.step(action.item())
            rr, map=None,None
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return

        if done:
            frame = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        # If our environment is keeping track of this for us (EpisodicLifeEnv) use that return instead.
        if "episode_return" in prior_info:
            # The episode_return will be None until the episode is done. We make it a NaN so we can still use the
            # numpy buffer.
            prior_return = prior_info["episode_return"]
            episode_return = torch.tensor(prior_return if prior_return is not None else np.nan)
            self.episode_return = episode_return
        if len(frame.shape)<=1:
            frame=torch.from_numpy(frame)
        elif len(frame.shape)==2:
            frame = torch.from_numpy(frame)
            #frame = torch.flatten(frame)[:201]#[:4]#frame[0:7, 0:7]
            #frame=frame[0:7, 0:7]
            frame=frame.unsqueeze(0)
            frame = frame.unsqueeze(0)
        else:
            frame = _format_frame(frame)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        if return_rr:
            return dict(
                frame=frame,
                reward=reward,
                done=done,
                episode_return=episode_return,
                episode_step=episode_step,
                last_action=action,
                rr=rr,
                map=map,
                pik=pik,
                match=match
            )
        else:
            return  dict(
                frame=frame,
                reward=reward,
                done=done,
                episode_return=episode_return,
                episode_step=episode_step,
                last_action=action,
            )

    def close(self):
        self.gym_env.close()
