import gym
from continual_rl.utils.env_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    TimeLimit,
    EpisodicLifeEnv,
    FireResetEnv,
    WarpFrame,
    ScaledFloatFrame,
    ClipRewardEnv,
    FrameStack,
)
from .image_task import ImageTask
from .cartpole_task import CartPoleTask
from .buglog_task import BuglogTask
import config_
def make_atari(env_id, max_episode_steps=None, full_action_space=False):
    env = gym.make(env_id, full_action_space=full_action_space)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


def get_single_atari_task(task_id, action_space_id, env_name, num_timesteps, max_episode_steps=None, full_action_space=False):
    """
    Wrap the task creation in a scope so the env_name in the lambda doesn't change out from under us.
    The atari max step default is 100k.
    """

    if 'CartPole' in env_name:
        return CartPoleTask(task_id=task_id, action_space_id=action_space_id, env_spec=env_name, num_timesteps=num_timesteps, time_batch_size=None, eval_mode=config_.eval_mode)
    if "bug_log" in env_name:
        if env_name[3]=='0':
            evalm=False
        else:
            evalm = True
        return BuglogTask(task_id=task_id, action_space_id=action_space_id, env_spec=env_name, num_timesteps=num_timesteps, time_batch_size=None, eval_mode=evalm)
        #return LTREnvV2(data_path=train_data_path, model_path="microsoft/codebert-base",
         #          tokenizer_path="microsoft/codebert-base", action_space_dim=31, report_count=100, max_len=512,
         #          use_gpu=False, caching=True, file_path=file_path, project_list=[project_name])
    # if "bug_log2" in env_name:
    #     return BuglogTask(task_id=task_id, action_space_id=action_space_id, env_spec=env_name,
    #                       num_timesteps=num_timesteps, time_batch_size=None, eval_mode=config_.eval_mode)
    #     #return LTREnvV2(data_path=train_data_path, model_path="microsoft/codebert-base",
    #     #           tokenizer_path="microsoft/codebert-base", action_space_dim=31, report_count=100, max_len=512,
    #     #           use_gpu=False, caching=True, file_path=file_path, project_list=[project_name])
    # if "bug_log3" in env_name:
    #     return BuglogTask(task_id=task_id, action_space_id=action_space_id, env_spec=env_name,
    #                       num_timesteps=num_timesteps, time_batch_size=None, eval_mode=config_.eval_mode)
    #     #return LTREnvV2(data_path=train_data_path, model_path="microsoft/codebert-base",
    #      #          tokenizer_path="microsoft/codebert-base", action_space_dim=31, report_count=100, max_len=512,
    #      #          use_gpu=False, caching=True, file_path=file_path, project_list=[project_name])
    return ImageTask(
        task_id=task_id,
        action_space_id=action_space_id,
        env_spec=lambda: wrap_deepmind(
            make_atari(env_name, max_episode_steps=max_episode_steps, full_action_space=full_action_space),
            clip_rewards=False,  # If policies need to clip rewards, they should handle it themselves
            frame_stack=False,  # Handled separately
            scale=False,
        ),
        num_timesteps=num_timesteps,
        time_batch_size=4,
        eval_mode=False,
        image_size=[84, 84],
        grayscale=True,
    )
