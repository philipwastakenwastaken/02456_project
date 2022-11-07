import torch
import numpy as np
import random

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

import os

from dqn_train import train_dq_model
from eval import eval_model
from model import QNetwork
from plotting import Plot

class Session:

    def __init__(self, config):
        # Retrieve data from config files
        session_params = config.session
        train_params = config.train
        model_params = config.model
        eval_params = config.eval
        env_params = config.environment

        # Setup seeds for predictability
        torch.manual_seed(session_params["seed"])
        random.seed(session_params['seed'])
        np.random.seed(session_params['seed'])

        path = os.path.join('models', 'dqnet.pt')
        self.model_path = os.path.join(get_original_cwd(), path)

        # Setup GPU
        if torch.cuda.is_available():
          self.dev = "cuda:0"
        else:
          self.dev = "cpu"

        self.model = QNetwork(n_inputs=model_params['n_inputs'],
                              n_outputs=model_params['n_outputs'],
                              learning_rate=model_params['learning_rate'])

        self.target_model = QNetwork(n_inputs=model_params['n_inputs'],
                                     n_outputs=model_params['n_outputs'],
                                     learning_rate=model_params['learning_rate'])

        if session_params['command'] == 'train':
            print('train!')
            self.train(train_params, env_params)
        elif session_params['command'] == 'evaluate':
            self.evaluate(eval_params, env_params)
        elif session_params['command'] == 'plot':
            self.plot(session_params, eval_params, env_params)
        else:
            raise Exception('Unknown command')

    def train(self, train_params, env_params):
        train_dq_model(self.dev,
                       train_params,
                       self.model,
                       self.target_model,
                       self.model_path,
                       env_params)

    def evaluate(self, eval_params, env_params):
        self.model.load_state_dict(torch.load(self.model_path))

        total_reward, i = eval_model(self.dev,
                                  eval_params,
                                  self.model,
                                  env_params)

    def plot(self, session_params, eval_params, env_params):
        self.model.load_state_dict(torch.load(self.model_path))
        PlotObject = Plot(session_params['numberOfReps'], self.dev,
                                  eval_params,
                                  self.model,
                                  env_params)
        PlotObject.plotTrainedModel()

    def setup_model(self):
        pass


@hydra.main(config_path="hparams/", config_name="default_config")
def objective(config: DictConfig):
    Session(config)


if __name__ == '__main__':
    objective()