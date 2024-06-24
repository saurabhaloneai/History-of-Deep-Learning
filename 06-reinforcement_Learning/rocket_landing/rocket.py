import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the Agent
class RocketLandingAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state).detach())
            target_f = self.model(state)
            target_f[0][action] = target
            loss = nn.MSELoss()(target_f, self.model(state))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training the agent with visualization
def train_agent(episodes, batch_size):
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = RocketLandingAgent(state_size, action_size)
    
    scores = []
    epsilons = []
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        scores.append(total_reward)
        epsilons.append(agent.epsilon)
        
        print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
        
        # Visualization every 100 episodes
        if (e + 1) % 100 == 0:
            plot_progress(e + 1, scores, epsilons)
    
    return agent, scores, epsilons

# Function to plot progress
def plot_progress(episode, scores, epsilons):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title(f'Score vs Episode (Episode {episode})')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(epsilons)
    plt.title(f'Epsilon vs Episode (Episode {episode})')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig(f'progress_episode_{episode}.png')
    plt.close()

# Function to visualize a single episode
def visualize_episode(agent, env):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        env.render()
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    
    env.close()
    print(f"Episode finished with score: {total_reward}")

# Train the agent
trained_agent, scores, epsilons = train_agent(episodes=1000, batch_size=64)

# Save the trained model
torch.save(trained_agent.model.state_dict(), 'rocket_landing_model.pth')

# Plot final results
plot_progress(1000, scores, epsilons)

# Visualize a single episode with the trained agent
env = gym.make('LunarLander-v2')
visualize_episode(trained_agent, env)