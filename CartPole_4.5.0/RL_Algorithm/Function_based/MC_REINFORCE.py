from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from collections import namedtuple, deque
import random
import matplotlib
import matplotlib.pyplot as plt
import os

class MC_REINFORCE_network(nn.Module):
    """
    Neural network for the MC_REINFORCE algorithm.
    
    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """

    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(MC_REINFORCE_network, self).__init__()
        # ========= put your code here ========= #
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, n_actions)
        # ====================================== #

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor representing action probabilities.
        """
        # ========= put your code here ========= #
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return F.softmax(x, dim=-1)
        # ====================================== #

class MC_REINFORCE(BaseAlgorithm):
    def __init__(
            self,
            device = None,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            n_observations: int = 4,
            hidden_dim: int = 64,
            dropout: float = 0.5,
            learning_rate: float = 0.01,
            discount_factor: float = 0.95,
    ) -> None:
        """
        Initialize the CartPole Agent.

        Args:
            learning_rate (float): The learning rate for updating Q-values.
            initial_epsilon (float): The initial exploration rate.
            epsilon_decay (float): The rate at which epsilon decays over time.
            final_epsilon (float): The final exploration rate.
            discount_factor (float, optional): The discount factor for future rewards. Defaults to 0.95.
        """     

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.LR = learning_rate

        self.policy_net = MC_REINFORCE_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)

        self.device = device
        self.steps_done = 0

        self.episode_durations = []

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        pass
        # ====================================== #

        super(MC_REINFORCE, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()
    
    def calculate_stepwise_returns(self, rewards):
        """
        Compute stepwise returns for the trajectory.

        Args:
            rewards (list): List of rewards obtained in the episode.
        
        Returns:
            Tensor: Normalized stepwise returns.
        """
        # ========= put your code here ========= #
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.discount_factor * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-9)
        returns = F.normalize(returns,dim=0)
        return returns
        # ====================================== #

    def generate_trajectory(self, env):
        """
        Generate a trajectory by interacting with the environment.

        Args:
            env: The environment object.
        
        Returns:
            Tuple: (episode_return, stepwise_returns, log_prob_actions, trajectory)
        """
        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Store state-action-reward history (list)
        # Store log probabilities of actions (list)
        # Store rewards at each step (list)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        obs,_ = env.reset()
        log_prob_actions = []
        rewards = []
        trajectory = []
        entropies = []
        episode_return = 0.0
        done = False
        timestep = 0
        # ====================================== #
        
        # ===== Collect trajectory through agent-environment interaction ===== #
        while not done:
            
            # Predict action from the policy network
            # ========= put your code here ========= #
            state = obs['policy']
            if torch.isnan(state).any():
                print("WARNING: State contains NaNs!")
            probs = self.policy_net(state)
            probs = torch.clamp(probs, min=1e-6, max=1.0) # avoid log(0) or NaN sampling
            dist = torch.distributions.Categorical(probs)
            action = dist.sample() #action idx
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            entropies.append(entropy)
            # print(probs)
            # ====================================== #

            # Execute action in the environment and observe next state and reward
            # ========= put your code here ========= #
            scaled_action = self.scale_action(action)
            # print(scaled_action)
            action_tensor = torch.tensor([[scaled_action]], dtype=torch.float32)
            # print(action_tensor)
            next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
            done = terminated or truncated
            # ====================================== #

            # Store action log probability reward and trajectory history
            # ========= put your code here ========= #
            log_prob_actions.append(log_prob)
            rewards.append(reward.item())
            trajectory.append((state.cpu().numpy(), action.item(), reward.item()))
            # ====================================== #
            
            # Update state
            # state = next_state
            obs = next_obs
            episode_return += reward.item()
            timestep += 1
            if done:
                self.plot_durations(timestep)
                break

        # ===== Stack log_prob_actions &  stepwise_returns ===== #
        # ========= put your code here ========= #
        stepwise_returns = self.calculate_stepwise_returns(rewards)
        # print(log_prob_actions)
        log_prob_actions = torch.stack(log_prob_actions)
        # print(log_prob_actions)
    
        return episode_return, stepwise_returns, log_prob_actions, trajectory,torch.stack(entropies),timestep
        # ====================================== #
    
    def calculate_loss(self, stepwise_returns, log_prob_actions,entropies):
        """
        Compute the loss for policy optimization.

        Args:
            stepwise_returns (Tensor): Stepwise returns for the trajectory.
            log_prob_actions (Tensor): Log probabilities of actions taken.
        
        Returns:
            Tensor: Computed loss.
        """
        # ========= put your code here ========= #
        # return -torch.sum(log_prob_actions * stepwise_returns) / len(stepwise_returns)
        # reinforce_loss = -torch.sum(log_prob_actions * stepwise_returns) / len(stepwise_returns)
        # entropy_bonus = 0.01 * torch.sum(entropies)  # Entropy weight can be tuned
        # return reinforce_loss - entropy_bonus
        loss = -sum(log_prob * ret for log_prob, ret in zip(log_prob_actions, stepwise_returns))
        loss /= len(stepwise_returns)
        return loss
        # ====================================== #

    def update_policy(self, stepwise_returns, log_prob_actions, entropies):
        """
        Update the policy using the calculated loss.

        Args:
            stepwise_returns (Tensor): Stepwise returns.
            log_prob_actions (Tensor): Log probabilities of actions taken.
        
        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        loss = self.calculate_loss(stepwise_returns, log_prob_actions, entropies)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        # ====================================== #
    
    def learn(self, env):
        """
        Train the agent on a single episode.

        Args:
            env: The environment to train in.
        
        Returns:
            Tuple: (episode_return, loss, trajectory)
        """
        # ========= put your code here ========= #
        self.policy_net.train()
        episode_return, stepwise_returns, log_prob_actions, trajectory, entropies,timestep = self.generate_trajectory(env)
        loss = self.update_policy(stepwise_returns, log_prob_actions,entropies)

        return episode_return, loss, timestep,trajectory
        # ====================================== #


    # Consider modifying this function to visualize other aspects of the training process.
    # ================================================================================== #
    def plot_durations(self, timestep=None, show_result=False):
        if timestep is not None:
            self.episode_durations.append(timestep)

        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        # if self.is_ipython:
        #     if not show_result:
        #         display.display(plt.gcf())
        #         display.clear_output(wait=True)
        #     else:
        #         display.display(plt.gcf())
    # ================================================================================== #
    def save_model(self, path, filename):

        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        torch.save(self.policy_net.state_dict(), full_path)
        print(f"Model saved to {full_path}")

    def load_model(self, path, filename):
    
        full_path = os.path.join(path, filename)
        self.policy_net.load_state_dict(torch.load(full_path, map_location=self.device))
        # self.policy_net.load_state_dict(self.policy_net.state_dict())  # Sync target with policy
        print(f"Model loaded from {full_path}")