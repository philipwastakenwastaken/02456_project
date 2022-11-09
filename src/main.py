import torch
import numpy as np
import random

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

import wandb

import os

from dqn_train import train_dq_model
from eval import eval_model
from model import QNetwork

class Session:

    def __init__(self, config):
        # Retrieve data from config files
        self.session_params = config.session
        self.train_params = config.train
        self.model_params = config.model
        self.eval_params = config.eval
        self.env_params = config.environment

        # Setup seeds for predictability
        torch.manual_seed(self.session_params["seed"])
        random.seed(self.session_params['seed'])
        np.random.seed(self.session_params['seed'])

        path = os.path.join('models', 'dqnet.pt')
        self.model_path = os.path.join(get_original_cwd(), path)

        # Setup GPU
        if torch.cuda.is_available():
          self.dev = "cuda:0"
        else:
          self.dev = "cpu"

        self.setup_wandb()

        self.model = QNetwork(n_inputs=self.model_params['n_inputs'],
                              n_outputs=self.model_params['n_outputs'],
                              learning_rate=self.model_params['learning_rate'])

        self.target_model = QNetwork(n_inputs=self.model_params['n_inputs'],
                                     n_outputs=self.model_params['n_outputs'],
                                     learning_rate=self.model_params['learning_rate'])

        if self.session_params['command'] == 'train':
            print('train!')
            self.train()
        elif self.session_params['command'] == 'evaluate':
            self.evaluate()
        else:
            raise Exception('Unknown command')

    def train(self):
        train_dq_model(self.dev,
                       self.train_params,
                       self.model,
                       self.target_model,
                       self.model_path,
                       self.env_params)

    def evaluate(self):
        self.model.load_state_dict(torch.load(self.model_path))

        total_reward = eval_model(self.dev,
                                  self.eval_params,
                                  self.model,
                                  self.env_params)

    def setup_model(self):
        pass

    def setup_wandb(self):
        self.use_wandb = self.session_params['use_wandb']
        try:
            key_path = os.path.join(get_original_cwd, 'wandb_api_key.txt')
            with open(key_path, encoding='utf-8') as f:
                key = f.read()
                if key == '':
                    raise Exception()
        except Exception:
            self.use_wandb = False

        if self.session_params['wandb_api_key'] != None:
            key = self.session_params['wandb_api_key']
            self.use_wandb = True

        print("wandb usage is:", self.use_wandb)
        self.logger = None

        if self.use_wandb and self.session_params['command'] == 'train':
            print("Using key: ", key)
            wandb.login(key=key)
            #self.logger = wandb.WandbLogger()

            wandb.init(project="02456_project", entity="philipwastaken")
            wandb.config.update = {
              "learning_rate": self.model_params.learning_rate,
              "num_episodes": self.train_params.num_episodes,
              "batch_size": self.train_params.batch_size,
              "gamma": self.train_params['gamma'],
              "val_freq": self.train_params['val_freq'],
              "tau": self.train_params['tau'],
              "replay_memory_capacity": self.train_params['replay_memory_capacity'],
              "prefill_memory": self.train_params['prefill_memory'],
              "episode_limit": self.train_params['episode_limit']
            }

            # Set a run name
            run_name = self.env_params['env_name'] + '-' + self.model_params['optim']
            run_name += '-' + wandb.run.name.split('-')[-1]
            run_name += '-' + 'lr=' + str(self.model_params['learning_rate'])
            wandb.run.name = run_name
            wandb.run.save()


@hydra.main(config_path="hparams/", config_name="default_config")
def objective(config: DictConfig):
    Session(config)


if __name__ == '__main__':
    objective()