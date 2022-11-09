import matplotlib.pyplot as plt
import numpy as np
from eval import  eval_model

class Plot():
    def __init__(self, iterations = None, dev = None, eval_params = None, dqnet = None, env_params = None, trainingResults = None):
        self.iterations = iterations
        self.dev = dev
        self.eval_params = eval_params
        self.dqnet = dqnet
        self.env_params = env_params
        self.trainingResults = trainingResults
        self.rewards = []

    def plotTrainedModel(self):
        for i in range(self.iterations):
            R, i = eval_model(self.dev, self.eval_params, self.dqnet, self.env_params, humanMode=False)
            self.rewards.append(R)
            print("Total reward in round {} is {}.".format(i, R))

        plt.plot(self.rewards)
        plt.xlabel('Round')
        plt.ylabel('Mean training reward')
        plt.show()

        print("Mean reward is {}".format(np.mean(self.rewards)))

    def plotTrainingProgress(self):
        Rewards = self.trainingResults[0]
        plt.plot(Rewards)
        plt.xlabel('Iteration')
        plt.ylabel('Mean training reward')
        plt.show()
