import torch
import warnings
warnings.filterwarnings("ignore")


def eval_model(dev, eval_params, dqnet, env):
    s = env.reset()
    HEIGHT = s.shape[0]
    WIDTH = s.shape[1]

    done = False
    frame = 1
    R = 0
    while not done:
        a = dqnet(torch.from_numpy(
            s.reshape((1, 1, HEIGHT, WIDTH))).float()).argmax().item()
        s, r, done, _ = env.step(a)

        R += r

        if done:
            print("Died at frame:", frame)
            return R, frame

        frame += 1

    print("Total reward", R)
    return R
