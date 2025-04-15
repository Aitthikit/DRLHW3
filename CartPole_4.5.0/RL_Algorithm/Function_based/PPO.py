import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from RL_Algorithm.RL_base_function import BaseAlgorithm
import matplotlib
import matplotlib.pyplot as plt

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=1e-4):
        """
        Actor network for policy approximation.

        Args:
            input_dim (int): Dimension of the state space.
            hidden_dim (int): Number of hidden units in layers.
            output_dim (int): Dimension of the action space.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(Actor, self).__init__()

        # ========= put your code here ========= #
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))  # Learned log std

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.init_weights()
        # ====================================== #

    def init_weights(self):
        """
        Initialize network weights using Xavier initialization for better convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier initialization
                nn.init.zeros_(m.bias)  # Initialize bias to 0

    def forward(self, state):
        """
        Forward pass for action selection.

        Args:
            state (Tensor): Current state of the environment.

        Returns:
            Tensor: Selected action values.
        """
        # ========= put your code here ========= #
        # x = self.net(state)
        # mean = self.mean(x)
        # std = self.log_std.exp()
        # return mean, std
        x = self.net(state)
        x = self.mean(x)
        x = torch.softmax(x, dim=-1)  # Softmax to get probability distribution
        return x
        # ====================================== #

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, learning_rate=1e-4):
        """
        Critic network for state-value approximation.

        Args:
            state_dim (int): Dimension of the state space.
            hidden_dim (int): Number of hidden units in layers.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, state):
        """
        Forward pass for state-value estimation.

        Args:
            state (Tensor): Current state of the environment.

        Returns:
            Tensor: Estimated V(s) value.
        """
        return self.net(state).squeeze(-1)

class PPO(BaseAlgorithm):
    def __init__(self, 
                device = None, 
                num_of_action: int = 2,
                action_range: list = [-2.5, 2.5],
                n_observations: int = 4,
                hidden_dim = 256,
                dropout = 0.05, 
                learning_rate: float = 0.01,
                tau: float = 0.005,
                discount_factor: float = 0.95,
                buffer_size: int = 256,
                batch_size: int = 1,
                ):
        """
        Actor-Critic algorithm implementation.

        Args:
            device (str): Device to run the model on ('cpu' or 'cuda').
            num_of_action (int, optional): Number of possible actions. Defaults to 2.
            action_range (list, optional): Range of action values. Defaults to [-2.5, 2.5].
            n_observations (int, optional): Number of observations in state. Defaults to 4.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
            tau (float, optional): Soft update parameter. Defaults to 0.005.
            discount_factor (float, optional): Discount factor for Q-learning. Defaults to 0.95.
            batch_size (int, optional): Size of training batches. Defaults to 1.
            buffer_size (int, optional): Replay buffer size. Defaults to 256.
        """
        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.device = device
        self.actor = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        self.actor_target = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        self.critic = Critic(n_observations,  hidden_dim, learning_rate).to(device)
        self.critic_target = Critic(n_observations,  hidden_dim, learning_rate).to(device)

        self.batch_size = batch_size
        self.tau = tau
        self.discount_factor = discount_factor

        self.update_target_networks(tau=1)  # initialize target networks
        self.episode_durations = []

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        pass
        # ====================================== #

        super(PPO, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

    def select_action(self, state, noise=0.0):
        """
        Selects an action based on the current policy with optional exploration noise.
        
        Args:
        state (Tensor): The current state of the environment.
        noise (float, optional): The standard deviation of noise for exploration. Defaults to 0.0.

        Returns:
            Tuple[Tensor, Tensor]: 
                - scaled_action: The final action after scaling.
                - clipped_action: The action before scaling but after noise adjustment.
        """
        # ========= put your code here ========= #
        # state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.actor(state)
        probs = torch.clamp(probs, min=1e-6, max=1.0) # avoid log(0) or NaN sampling
        dist = torch.distributions.Categorical(probs)
        # print(dist)
        action = dist.sample()
        # print(action)
        # action_clipped = action.clamp(*self.action_range)
        log_prob = dist.log_prob(action).sum(dim=-1)
        # print(action_clipped,log_prob)
        return action, log_prob
        # ====================================== #
    
    def generate_sample(self, batch_size):
        """
        Generates a batch sample from memory for training.

        Returns:
            Tuple: A tuple containing:
                - state_batch (Tensor): The batch of current states.
                - action_batch (Tensor): The batch of actions taken.
                - reward_batch (Tensor): The batch of rewards received.
                - next_state_batch (Tensor): The batch of next states received.
                - done_batch (Tensor): The batch of dones received.
        """
        # Ensure there are enough samples in memory before proceeding
        # ========= put your code here ========= #
        # Sample a batch from memory
        batch = self.memory.sample()
        # ====================================== #
        
        # Sample a batch from memory
        # ========= put your code here ========= #
        if len(self.memory) < batch_size:
            return None
        return self.memory.sample()
        # ====================================== #

    def calculate_loss(self, states, actions, rewards, next_states, dones, old_log_probs, advantages):
        """
        Computes the loss for policy optimization.

        Args:
            - states (Tensor): The batch of current states.
            - actions (Tensor): The batch of actions taken.
            - rewards (Tensor): The batch of rewards received.
            - next_states (Tensor): The batch of next states received.
            - dones (Tensor): The batch of dones received.

        Returns:
            Tensor: Computed critic & actor loss.
        """
        # ========= put your code here ========= #
        # Update Critic
        values = self.critic(states)
        critic_loss = mse_loss(values, rewards)
        # Gradient clipping for critic

        # Update Actor
        logits = self.actor(states)                        # shape: (batch_size, num_actions)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)             # shape: (batch_size,)

        # Gradient clipping for actor
        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
        actor_loss = -torch.min(surrogate1, surrogate2).mean()

        return actor_loss, critic_loss
        # ====================================== #
    def safe_standardize(self,tensor, eps=1e-6):
        std = tensor.std()
        return (tensor - tensor.mean()) / (std + eps) if std > 0 else tensor - tensor.mean()

    def update_policy(self):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        sample = self.generate_sample(self.batch_size)
        if sample is None:
            return
        states, actions, rewards, next_states, dones = sample

        actions = actions.long()

        with torch.no_grad():
            # Estimate state values
            values = self.critic(states)
            next_values = self.critic(next_states)
            td_target = rewards + self.discount_factor * next_values * (~dones)
            advantages = td_target - values
            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
            advantages = self.safe_standardize(advantages)

            # Get action probabilities from the actor
            action_probs = self.actor(states)
            dist = torch.distributions.Categorical(action_probs)
            old_log_probs = dist.log_prob(actions)

        # Normalize rewards (optional but often helpful)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        rewards = self.safe_standardize(rewards)

        # Compute critic and actor loss
        actor_loss, critic_loss = self.calculate_loss(states, actions, rewards, next_states, dones, old_log_probs, advantages)
        
        # Backpropagate and update critic network parameters
        
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        # Backpropagate and update actor network parameters
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        # ====================================== #


    def update_target_networks(self, tau=None):
        """
        Perform soft update of target networks using Polyak averaging.

        Args:
            tau (float, optional): Update rate. Defaults to self.tau.
        """
        # ========= put your code here ========= #
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        # ====================================== #

    def learn(self, env, max_steps, num_agents, noise_scale=0.1, noise_decay=0.99):
        """
        Train the agent on a single step.

        Args:
            env: The environment in which the agent interacts.
            max_steps (int): Maximum number of steps per episode.
            num_agents (int): Number of agents in the environment.
            noise_scale (float, optional): Initial exploration noise level. Defaults to 0.1.
            noise_decay (float, optional): Factor by which noise decreases per step. Defaults to 0.99.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        obs,_ = env.reset()
        total_reward = 0
        # ====================================== #

        for step in range(max_steps):
            # Predict action from the policy network
            # ========= put your code here ========= #
            state = obs['policy']
            action, log_prob = self.select_action(state)
            # ====================================== #

            # Execute action in the environment and observe next state and reward
            # ========= put your code here ========= #
            # print(action)
            scaled_action = self.scale_action(action)
            action_tensor = torch.tensor([[scaled_action]], dtype=torch.float32)
            # next_state, reward, done, _ = env.step(action.cpu().numpy())
            print(action_tensor)
            next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
            next_state = next_obs['policy']
            done = terminated or truncated
            # ====================================== #

            # Store the transition in memory
            # ========= put your code here ========= #
            self.memory.add(state, action, reward, next_state, done)
            # Parallel Agents Training
            if num_agents > 1:
                pass
            # Single Agent Training
            else:
                pass
            # ====================================== #

            # Update state
            total_reward += reward
            obs = next_obs
            # Decay the noise to gradually shift from exploration to exploitation
            if done:
                self.plot_durations(step)
                break

            # Perform one step of the optimization (on the policy network)
        self.update_policy()

            # Update target networks
        self.update_target_networks()

        return total_reward,step

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