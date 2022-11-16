import torch
import numpy as np
import random

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

import wandb

import os
import datetime
import glob

from dqn_train import train_dq_model
from eval import eval_model
from model import QNetwork
from plotting import Plot
from env_factory import make_env


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

        self.setup_device()
        self.setup_wandb()
        self.setup_model()

        if self.session_params['command'] == 'train':
            self.train()
        elif self.session_params['command'] == 'evaluate':
            self.evaluate()
        elif self.session_params['command'] == 'plot':
            self.plot()
        else:
            raise Exception('Unknown command')

    def train(self):
        env = make_env(self.env_params, self.model_params)
        rewards, lengths, losses, epsilons = train_dq_model(self.dev,
                                                            self.train_params,
                                                            self.model,
                                                            self.target_model,
                                                            self.model_path,
                                                            self.use_wandb,
                                                            self.checkpoint,
                                                            env)

        # On HPC cluster we don't want to render plots.
        if self.session_params['show_plots']:
            PlotObject = Plot(trainingResults=(
                rewards, lengths, losses, epsilons))
            PlotObject.plotTrainingProgress()

    def plot(self):
        PlotObject = Plot(self.session_params['numberOfReps'],
                          self.dev,
                          self.eval_params,
                          self.model,
                          self.env_params)
        PlotObject.plotTrainedModel()

    def evaluate(self):
        human_mode = True
        if human_mode:
            env = make_env(self.env_params, self.model_params,
                           render_mode='human')
        else:
            env = make_env(self.env_params, self.model_params)

        self.dqnet.to(torch.device(self.dev))
        total_reward, i = eval_model(self.model, env, self.dev)

    def setup_device(self):
        dev_option = self.session_params['device']

        # Default to GPU if available
        if dev_option == 'auto':
            if torch.cuda.is_available():
                self.dev = "cuda:0"
            else:
                self.dev = "cpu"
        elif dev_option == 'cpu':
            self.dev = "cpu"
        elif dev_option == 'gpu':
            if torch.cuda.is_available():
                self.dev = "cuda:0"
            else:
                raise Exception("Cuda is not available, GPU cannot be used")
        else:
            raise Exception("Unknown device argument")

        print(f'< device: {self.dev} >')


    def load_checkpoint(self):
        checkpoint = torch.load(self.model_path)
        self.checkpoint = checkpoint
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['model_state_dict'])
        self.model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    def setup_model(self):
        self.checkpoint = None


        self.model = QNetwork(n_inputs=self.model_params['n_inputs'],
                              n_outputs=self.model_params['n_outputs'],
                              learning_rate=self.model_params['learning_rate'])

        self.target_model = QNetwork(n_inputs=self.model_params['n_inputs'],
                                     n_outputs=self.model_params['n_outputs'],
                                     learning_rate=self.model_params['learning_rate'])

        config_model_path = self.model_params['model_path']
        config_model_path_set = config_model_path != ''
        is_train = self.session_params['command'] == 'train'

        # Case 1: no model path set and we want to train. Training a fresh model with a newly generated name.
        if not config_model_path_set and is_train:
            timeNow = str(datetime.datetime.now()).replace(" ", "_")
            timeNow = timeNow.replace(".", "D")
            timeNow = timeNow.replace("-", "_")
            timeNow = timeNow.replace(":", "DD")

            subpath = self.model_params['wrapper'] + \
                '_' + timeNow + '_dqnet.pt'
            self.model_path = os.path.join(
                get_original_cwd(), 'models', subpath)

        # Case 2: model path is set and we want to train. Load weights from model path and continue training
        #         this model.
        if config_model_path_set and is_train:
            self.model_path = os.path.join(
                get_original_cwd(), 'models', config_model_path)
            self.load_checkpoint()

        # Case 3: model path is set and we want to run in plot or evaluate mode.
        #         Load model from path and continue.
        if config_model_path_set and not is_train:
            self.model_path = os.path.join(
                get_original_cwd(), 'models', config_model_path)
            self.load_checkpoint()

        # Case 4: model path is not set and we want to run in plot or evaluate mode.
        #         As a fallback, simply load the name of the most recently created model.
        #         WARNING: this is based on file modification date, NOT on the timestamp in the name.
        #         If no models are found in the models/ folder, an exception is raised.
        if not config_model_path_set and not is_train:
            path = os.path.join(get_original_cwd(), 'models', '*_dqnet.pt')
            names = [x for x in glob.glob(path)]
            names.sort(key=os.path.getmtime, reverse=True)
            print(names)

            if len(names) == 0:
                raise Exception(
                    'No model path set with no fallback model found')

            self.model_path = names[0]
            self.load_checkpoint()

    def setup_wandb(self):
        self.use_wandb = self.session_params['use_wandb']

        if self.session_params['wandb_api_key'] != None:
            key = self.session_params['wandb_api_key']

        print(f'< wandb: {self.use_wandb}')

        if self.use_wandb and self.session_params['command'] == 'train':
            print("Using key: ", key)
            wandb.login(key=key)

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

            #wandb.define_metric("validate/step")
            #wandb.define_metric("validate/*", step_metric="validate/step")

            # Set a run name
            run_name = self.model_params['wrapper']
            run_name += '-' + wandb.run.name.split('-')[-1]
            run_name += '-' + 'batch=' + str(self.train_params['batch_size'])
            run_name += '-' + 'lr=' + str(self.model_params['learning_rate'])
            run_name += '-' + 'eps=' + str(self.train_params['num_episodes'])
            run_name += '-' + 'gamma=' + str(self.train_params['gamma'])
            wandb.run.name = run_name
            wandb.run.save()


@hydra.main(config_path="hparams/", config_name="default_config")
def objective(config: DictConfig):
    Session(config)


if __name__ == '__main__':
    objective()
