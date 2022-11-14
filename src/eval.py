import torch
import warnings
warnings.filterwarnings("ignore")

def eval_model(dev, eval_params, dqnet, env):
    s = env.reset()
    HEIGHT = s.shape[0]
    WIDTH = s.shape[1]

    R = 0
    for i in range(2000):
        a = dqnet(torch.from_numpy(s.reshape((1,1,HEIGHT,WIDTH))).float()).argmax().item()
        s, r, done, _ = env.step(a)

        R += r

        if done:
            print("Died at frame:",i)
            return R, i

    print("Total reward", R)
    return R
