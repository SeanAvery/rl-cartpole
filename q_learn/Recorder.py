import os
import gym
from gym import wrappers


class Recorder():
    def __init__(self):
        self.check_path()

    def check_path(self):
        if not os.path.exists('./simulations'):
            os.mkdir('./simulations')

    def wrap_env(self, env):
        random_id = os.urandom(4)
        return wrappers.Monitor(env, './simulations/{}'.format(random_id))



if __name__ == '__main__':
    recorder = Recorder()
    env_temp = gym.make('CartPole-v0')
    env = recorder.wrap_env(env_temp)
    env.reset()
    for i in range(10):
        env.render()
        state, reward, done, _ = env.step(0)

        if done:
            break

    env.close
