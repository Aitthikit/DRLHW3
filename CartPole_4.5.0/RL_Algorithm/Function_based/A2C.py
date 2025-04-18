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
import os



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
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
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
        x = self.net(state)
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

class A2C(BaseAlgorithm):
    def __init__(self, 
                device = None, 
                num_of_action: int = 2,
                action_range: list = [-2.5, 2.5],
                n_observations: int = 4,
                hidden_dim = 256,
                learning_rate: float = 0.01,
                discount_factor: float = 0.95,
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
        self.critic = Critic(n_observations,  hidden_dim, learning_rate).to(device)

        self.discount_factor = discount_factor
        self.episode_durations = []

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        pass
        # ====================================== #

        super(A2C, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
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
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
        # ====================================== #
    

    def calculate_loss(self, states, actions, rewards, next_states, dones):
        # Value estimates
        values = self.critic(states)
        next_values = self.critic(next_states)

        # TD target
        td_target = rewards + self.discount_factor * next_values
        advantages = td_target - values

        # Actor Loss
        logits = self.actor(states)
        dist = torch.distributions.Categorical(logits)
        log_probs = dist.log_prob(actions)
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Critic Loss (MSE between value and target)
        critic_loss = (advantages ** 2).mean()

        return actor_loss, critic_loss
        # ====================================== #
    def safe_standardize(self,tensor, eps=1e-6):
        std = tensor.std()
        return (tensor - tensor.mean()) / (std + eps) if std > 0 else tensor - tensor.mean()


    def update_policy(self,state, action, reward, next_state, done):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        action = action.long()

        actor_loss, critic_loss = self.calculate_loss(state, action, reward, next_state, done)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        return actor_loss, critic_loss

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
        total_actorloss = 0
        total_criticloss = 0
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
            # print(action_tensor)
            next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
            next_state = next_obs['policy']
            done = terminated or truncated
            # ====================================== #

            # Store the transition in memory
            # ========= put your code here ========= #
            actor_loss, critic_loss = self.update_policy(state, action, reward, next_state, done)
            # ====================================== #

            # Update state
            total_actorloss += actor_loss
            total_criticloss += critic_loss
            total_reward += reward
            obs = next_obs
            # Decay the noise to gradually shift from exploration to exploitation
            if done:
                self.plot_durations(step)
                break

            # Perform one step of the optimization (on the policy network)
        

        return total_reward,step,actor_loss, critic_loss

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
        """
        Save the actor and critic network weights to a single file.

        Args:
            path (str): Directory to save the model weights.
            filename (str): Filename for the saved weights.
        """
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, full_path)

        print(f"Actor and Critic models saved to {full_path}")

    def load_model(self, path, filename):
        """
        Load the actor and critic network weights from a file.

        Args:
            path (str): Directory where the model weights are saved.
            filename (str): Filename of the saved weights.
        """
        full_path = os.path.join(path, filename)
        
        checkpoint = torch.load(full_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

        print(f"Actor and Critic models loaded from {full_path}")