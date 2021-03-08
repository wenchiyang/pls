import gym
import relenvs

ENV_NAME = 'Blockworld-v0'
env = gym.make(ENV_NAME)

# Initial State
print(env.render())

# Make and action
_, reward, _, _ = env.step("move(7,9)")


print(env.render())
print("Reward", reward)

