import gymnasium as gym
import torch
import time

# policy = torch.load("good_lunar_policy.pt")
policy = torch.load("latest_model.pt")

def select_action(policy, state):
	soft = torch.softmax(policy(state), dim=1)
	dist = torch.distributions.Categorical(soft)
	return dist.sample()



env = gym.make("LunarLander-v2", render_mode="human")
env.action_space.seed(42)
observation, info = env.reset(seed=42)

while True:
    action = select_action(policy, torch.from_numpy(observation).reshape(1,-1)).item()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        try:
            policy = torch.load("latest_model.pt")
        except:
            print(f"Failed to update policy: {time.time()}")
        observation, info = env.reset()
env.close()

