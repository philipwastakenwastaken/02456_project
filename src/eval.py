import torch
import warnings
warnings.filterwarnings("ignore")


def eval_model(dqnet, env, dev):
    s = env.reset()
    HEIGHT = s.shape[0]
    WIDTH = s.shape[1]

    done = False
    frame = 1
    R = 0
    while not done:
        a = dqnet(torch.from_numpy(s.reshape((1, 1, HEIGHT, WIDTH))).float().to(
            torch.device(dev))).argmax().item()
        s, r, done, _ = env.step(a)

        R += r
        frame += 1

    return R, frame
