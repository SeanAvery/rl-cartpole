import os
import codecs
import gym
from gym import wrappers
from moviepy.editor import *
from time import sleep

class Recorder():
    def __init__(self):
        self.check_path()

    def check_path(self):
        if not os.path.exists('./simulations'):
            os.mkdir('./simulations')

    def wrap_env(self, env):
        self.random_id = codecs.encode(os.urandom(4), 'hex').decode()
        return wrappers.Monitor(env, './simulations/{}'.format(self.random_id))

    def make_gif(self):
        video_file = self.get_mp4_file()
        print('video_file', video_file)
        sleep(1)
        self.check_file_size(video_file)
        clip = (VideoFileClip('./simulations/{0}/{1}'.format(self.random_id, video_file)))
        clip.write_gif('./simulations/{0}/sim.gif'.format(self.random_id))

    def get_mp4_file(self):
        files = os.listdir('./simulations/{0}/'.format(self.random_id))
        for file_name in files:
            if file_name[-3:] == 'mp4':
                return file_name
        raise Exception('no video file available')

    def check_file_size(self, video_file):
        stat_info = os.stat('./simulations/{0}/{1}'.format(self.random_id, video_file))
        print('size', stat_info.st_size)

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
    env._close_video_recorder()
    recorder.make_gif()
