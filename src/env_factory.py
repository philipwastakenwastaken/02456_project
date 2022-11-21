import gym
from env_wrapper import *


def make_env(env_params, model_params, render_mode=None):
    env_name = env_params['env_name']

    if env_name == 'ALE/Asterix-v5':
        return make_asterix(env_params, render_mode)
    elif env_name == 'ALE/MsPacman-v5':
        return make_mspacman(env_params, model_params, render_mode)

    raise Exception("Unknown environment name")


def make_asterix(env_params, render_mode):
    env = gym.make(env_params['env_name'],
                   full_action_space=False,
                   obs_type='grayscale',
                   render_mode=render_mode)
    return env


def make_mspacman(env_params, model_params, render_mode):
    env = gym.make(env_params['env_name'],
                   full_action_space=False,
                   obs_type='grayscale',
                   render_mode=render_mode)
    wrapper = model_params['wrapper']

    if wrapper == 'gray':
        env = NopWrapper(env)

    elif wrapper == 'crop':
        env = CropWrapper(env)

    elif wrapper == 'scale120':
        env = Scale120Wrapper(env)

    elif wrapper == 'scale84':
        env = Scale84Wrapper(env)

    elif wrapper == 'scale72':
        env = Scale72Wrapper(env)

    return env
