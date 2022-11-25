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
        return NopWrapper(env)

    elif wrapper == 'crop':
        return CropWrapper(env)

    elif wrapper == 'scale120':
        return Scale120Wrapper(env)

    elif wrapper == 'scale84':
        return Scale84Wrapper(env)

    elif wrapper == 'scale72':
        return Scale72Wrapper(env)
    
    elif wrapper == 'scale60':
        return Scale60Wrapper(env)

    elif wrapper == 'scale48':
        return Scale48Wrapper(env)

    print("Wrapper init error!")
    return