import torch
import numpy as np
import random

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


from dqn_train import train_dq_model
from model import QNetwork

class Session:

    def __init__(self, config):
        # Retrieve data from config files
        session_params = config.session
        train_params = config.train
        model_params = config.model
        #eval_params = config.evaluate

        # Setup seeds for predictability
        torch.manual_seed(session_params["seed"])
        random.seed(session_params['seed'])
        np.random.seed(session_params['seed'])

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
            self.train(train_params)
        elif session_params['command'] == 'evaluate':
            self.evaluate()
        else:
            raise Exception('Unknown command')

    def train(self, train_params):
        train_dq_model(self.dev, train_params, self.model, self.target_model)

    def evaluate(self):
        pass

    def setup_model(self):
        pass




@hydra.main(config_path="hparams/", config_name="default_config")
def objective(config: DictConfig):
    sess = Session(config)
    return 0


if __name__ == '__main__':
    objective()
