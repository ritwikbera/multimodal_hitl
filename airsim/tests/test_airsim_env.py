import gym
import airsim_env
import time
from utils_airsim import JoystickAgent

# create env to be tested
env = gym.make(
    "MultimodalAirSimMountains-v0",
    n_steps=200,
    exp_name='test_env',
    custom_command='FLY AROUND')

# test env with joystick controls
agent = JoystickAgent()

# loop to test reset, observation and action space
for i_episode in range(3):
    observation = env.reset()
    for t in range(env.n_steps):
        print(f'episode {i_episode}, step {t}')
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

# close everything
env.close()