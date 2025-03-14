import gym
import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import MOVEMENT

import math
import random
from PIL import Image, ImageOps
import numpy as np
from collections import namedtuple, deque
from itertools import count
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import os

#os.environ['DISPLAY'] = '10.212.3.113:10.0'

# Checking if GPU Resources are available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN-BASED DQN Neural Network 
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Calculate size of the output of the last conv layer
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        
        # Linear layer
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
    #DQN
    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define the game screen
def preprocess_screen(screen):
    # Convert to grayscale
    img = Image.fromarray(screen)
    grayscale_img = ImageOps.grayscale(img)
    
    # Resize image
    resized_img = grayscale_img.resize((160, 144), Image.Resampling.LANCZOS)
    
    # Convert back to array and normalize
    resized_screen = np.array(resized_img) / 255.0
    
    # Add a batch dimension
    normalized_screen = np.expand_dims(resized_screen, axis=0)
    
    return normalized_screen

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, death_penalty=-40, line_clear_bonus=100):
        super().__init__(env)
        self.death_penalty = death_penalty
        self.line_clear_bonus = line_clear_bonus
            
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        if 'number_of_lines' in info:
            lines_cleared = info['number_of_lines']
            if lines_cleared > 0:
                reward += self.line_clear_bonus * lines_cleared

        if done:
            reward += self.death_penalty
        return state, reward, done, info

# Initialise the Tetris environment
env = gym_tetris.make('TetrisA-v2')
env = JoypadSpace(env, MOVEMENT)
env = CustomRewardWrapper(env, death_penalty=-40, line_clear_bonus=100)

# Episode amount depending on GPU Resources are available or not
if torch.cuda.is_available():
    num_episodes = 1000
else:
    num_episodes = 50

# Epsilion values
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 850
steps_done = 0


# Hyperparameters
LEARNING_RATE = 0.006
GAMMA = 0.85
BATCH_SIZE = 256
TARGET_UPDATE = 20

# Start the game
state = env.reset()
#env.render()

info = {}

init_screen = preprocess_screen(env.reset())
_, screen_height, screen_width = init_screen.shape
state = init_screen

# Number of actions
n_actions = env.action_space.n

# Initialise policy and target networks
policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
policy_net.train()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Target network is not trained

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

memory = ReplayMemory(50000)

def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)
    
def load_checkpoint(filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    policy_net.load_state_dict(checkpoint['policy_state_dict'])
    target_net.load_state_dict(checkpoint['target_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    steps_done = checkpoint['steps_done']
    return steps_done

def select_action(state, steps_done):
    global epsilon_end, epsilon_start, epsilon_decay, policy_net, n_actions
    sample = random.random()
    eps_threshold = epsilon_end + (epsilon_start - epsilon_end) * \
                    math.exp(-1. * steps_done / epsilon_decay)
    if sample > eps_threshold:
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device).float()
            return policy_net(state_tensor).max(1)[1].view(1, 1), eps_threshold
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), eps_threshold

def push_to_memory(state, action, next_state, reward):
    state_tensor = torch.from_numpy(state).float().to(device).unsqueeze(0)
    next_state_tensor = torch.from_numpy(next_state).float().to(device).unsqueeze(0) if next_state is not None else None
    action_tensor = torch.tensor([[action]], device=device, dtype=torch.long)
    reward_tensor = torch.tensor([reward], device=device)

    memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
        
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
 

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
# Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

# Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

# Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
# In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

 # Log the loss value
    print('Loss:', loss.item())

score_per_episode = []
lines_per_episode = []

# Parameters for checkpointing
checkpoint_path = "tetris_checkpoint.pth"
save_agent = 25  

# Check if a checkpoint exists and load it
if os.path.isfile(checkpoint_path):
    steps_done = load_checkpoint(checkpoint_path)
    print("Loaded checkpoint from:", checkpoint_path)
else:
    steps_done = 0
    print("No checkpoint found. Starting from scratch.")
    
#Training Loop
try:
    # Loop over each episode
    for i_episode in range(num_episodes):
        # Reset environment and initialize state at the start of each episode
        state = env.reset()
        state = preprocess_screen(env.render(mode='rgb_array'))
        episode_score = 0 
        episode_lines_cleared = 0 

        for t in count():
            env.render()
            # Select and perform an action
            action, eps_threshold = select_action(state, steps_done)
            steps_done += 1
            
            # Execute the selected action and observe new state and reward
            next_state, reward, done, info = env.step(action.item())
            
            #reward + action check
            print(f"Episode {i_episode}, Action: {action.item()}, Reward: {reward}")
            
            # Update the total score and lines cleared for the episode
            episode_score = info['score']
            episode_lines_cleared = info['number_of_lines']
            
            # Process the new state
            next_state = preprocess_screen(env.render(mode='rgb_array')) if not done else None
            
            # Convert reward to tensor and save transition in memory
            reward = torch.tensor([reward], device=device)
            push_to_memory(state, action.item(), next_state, reward)
            
            state = next_state if next_state is not None else preprocess_screen(env.reset())

            # Perform optimization step
            optimize_model()
            if t % 500 == 0:  
                clear_output(wait=True)
                
            if done:
                score_per_episode.append(episode_score)
                lines_per_episode.append(episode_lines_cleared) 
                print(f"Episode {i_episode} - Score: {episode_score}")
                break  # Exit the current episode loop to start a new episode

        #Update the target network after X episodes
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if i_episode % save_agent == 0 or i_episode == num_episodes-1:
            save_checkpoint({
                'policy_state_dict': policy_net.state_dict(),
                'target_state_dict': target_net.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'steps_done': steps_done
            }, checkpoint_path)
            print(f"Checkpoint saved at episode {i_episode}")
            
        

except Exception as e:
    print(f"An exception occurred: {e}")
    
except KeyboardInterrupt:
    print("Training interrupted. Saving current model.")
    save_checkpoint({
        'policy_state_dict': policy_net.state_dict(),
        'target_state_dict': target_net.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'steps_done': steps_done
    }, checkpoint_path)

finally:
    env.close()

 # Plotting Score per Episode
plt.figure(figsize=(12, 6))
plt.plot(score_per_episode, label='Score per Episode')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Score achieved per episode')
plt.legend()
plt.show()

 # Plotting Lines per Episode
plt.figure(figsize=(12, 6))
plt.plot(lines_per_episode, label='Lines per Episode')
plt.xlabel('Episode')
plt.ylabel('Lines Cleared')
plt.title('Lines cleared per episode')
plt.legend()
plt.show()

@misc{gym-tetris,
  author = {Christian Kauten},
  howpublished = {GitHub},
  title = {{Tetris (NES)} for {OpenAI Gym}},
  URL = {https://github.com/Kautenja/gym-tetris},
  year = {2019},
}