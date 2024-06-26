import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, max_length, hidden_size, num_heads, num_layers):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.state_encoder = nn.Linear(state_dim, hidden_size)
        self.action_encoder = nn.Linear(action_dim, hidden_size)
        self.return_encoder = nn.Linear(1, hidden_size)

        self.embed_timestep = nn.Embedding(max_length, hidden_size)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads),
            num_layers=num_layers
        )

        self.action_predictor = nn.Linear(hidden_size, action_dim)

    def forward(self, states, actions, returns_to_go, timesteps):
        batch_size, seq_length = states.shape[0], states.shape[1]

        state_embeddings = self.state_encoder(states)
        action_embeddings = self.action_encoder(actions)
        returns_embeddings = self.return_encoder(returns_to_go.unsqueeze(-1))

        time_embeddings = self.embed_timestep(timesteps)

        sequence = torch.stack(
            [returns_embeddings, state_embeddings, action_embeddings], dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        
        sequence = sequence + time_embeddings

        transformer_outputs = self.transformer(sequence)

        action_preds = self.action_predictor(transformer_outputs[:, 1::3])

        return action_preds

# Function to collect trajectories
def collect_trajectories(env, num_trajectories):
    trajectories = []
    for _ in range(num_trajectories):
        state = env.reset()
        done = False
        trajectory = []
        total_reward = 0
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
            total_reward += reward
        trajectories.append((trajectory, total_reward))
    return trajectories

# Function to prepare data for training
def prepare_data(trajectories, max_length):
    states, actions, returns_to_go, timesteps = [], [], [], []
    for trajectory, total_reward in trajectories:
        traj_states, traj_actions, traj_returns = [], [], []
        for t, (s, a, r) in enumerate(trajectory):
            traj_states.append(s)
            traj_actions.append(a)
            traj_returns.append(total_reward - sum(r for _, _, r in trajectory[:t]))
        
        # Pad trajectories to max_length
        traj_length = len(traj_states)
        if traj_length < max_length:
            padding = max_length - traj_length
            traj_states += [np.zeros_like(traj_states[0])] * padding
            traj_actions += [np.zeros_like(traj_actions[0])] * padding
            traj_returns += [0] * padding
        else:
            traj_states = traj_states[:max_length]
            traj_actions = traj_actions[:max_length]
            traj_returns = traj_returns[:max_length]
        
        states.append(traj_states)
        actions.append(traj_actions)
        returns_to_go.append(traj_returns)
        timesteps.append(list(range(max_length)))
    
    return (torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(returns_to_go),
            torch.LongTensor(timesteps))

# Training loop
def train(model, env, num_epochs, batch_size, lr, num_trajectories, max_length):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(num_epochs):
        trajectories = collect_trajectories(env, num_trajectories)
        states, actions, returns_to_go, timesteps = prepare_data(trajectories, max_length)
        
        for i in range(0, len(states), batch_size):
            batch_states = states[i:i+batch_size]
            batch_actions = actions[i:i+batch_size]
            batch_returns = returns_to_go[i:i+batch_size]
            batch_timesteps = timesteps[i:i+batch_size]
            
            action_preds = model(batch_states, batch_actions, batch_returns, batch_timesteps)
            loss = loss_fn(action_preds, batch_actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Evaluation function
def evaluate(model, env, max_length):
    state = env.reset()
    done = False
    total_reward = 0
    states, actions, returns_to_go, timesteps = [], [], [], []
    
    for t in range(max_length):
        if done:
            break
        
        states.append(state)
        returns_to_go.append(total_reward)
        timesteps.append(t)
        
        if len(states) < max_length:
            actions.append(np.zeros(env.action_space.shape[0]))
        
        model_input = prepare_data([(list(zip(states, actions, returns_to_go)), total_reward)], max_length)
        action_pred = model(*[tensor.unsqueeze(0) for tensor in model_input]).squeeze(0)[-1]
        
        action = action_pred.detach().numpy()
        next_state, reward, done, _ = env.step(action)
        
        state = next_state
        total_reward += reward
        
    return total_reward


if __name__ == "__main__":
    env = gym.make("LunarLanderContinuous-v2")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_length = 1000
    hidden_size = 128
    num_heads = 8
    num_layers = 3
    
    model = DecisionTransformer(state_dim, action_dim, max_length, hidden_size, num_heads, num_layers)
    
    # Training parameters
    num_epochs = 50
    batch_size = 64
    lr = 1e-4
    num_trajectories = 100
    
    # Train the model
    train(model, env, num_epochs, batch_size, lr, num_trajectories, max_length)
    
    # Evaluate the model
    num_eval_episodes = 10
    eval_rewards = [evaluate(model, env, max_length) for _ in range(num_eval_episodes)]
    
    print(f"Average evaluation reward: {np.mean(eval_rewards):.2f}")
    
    # Visualize a single episode
    state = env.reset()
    done = False
    total_reward = 0
    states, actions, returns_to_go, timesteps = [], [], [], []
    
    while not done:
        env.render()
        states.append(state)
        returns_to_go.append(total_reward)
        timesteps.append(len(states) - 1)
        
        if len(states) < max_length:
            actions.append(np.zeros(env.action_space.shape[0]))
        
        model_input = prepare_data([(list(zip(states, actions, returns_to_go)), total_reward)], max_length)
        action_pred = model(*[tensor.unsqueeze(0) for tensor in model_input]).squeeze(0)[-1]
        
        action = action_pred.detach().numpy()
        next_state, reward, done, _ = env.step(action)
        
        state = next_state
        total_reward += reward
    
    env.close()
    
    print(f"Visualization episode reward: {total_reward:.2f}")