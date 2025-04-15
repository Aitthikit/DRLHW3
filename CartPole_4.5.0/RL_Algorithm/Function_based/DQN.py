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

class QNetwork(nn.Module):
    def __init__(self, input_dim : int, hidden_dim: int, output_dim: int, learning_rate=1e-4,dropout=0.0):
        """
        Actor network for policy approximation.

        Args:
            input_dim (int): Dimension of the state space.(State_dim)
            hidden_dim (int): Number of hidden units in layers.
            output_dim (int): Dimension of the action space.(Actions_dim)
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(QNetwork, self).__init__()

        # ========= put your code here ========= #
        self.init_weights()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
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
        x = torch.relu(self.fc1(state))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)
        # ====================================== #

class DQN(BaseAlgorithm):
    def __init__(self, 
                device = None, 
                num_of_action: int = 2,
                action_range: list = [-2.5, 2.5],
                n_observations: int = 4,
                hidden_dim = 256,
                dropout = 0.05, 
                learning_rate: float = 0.01,
                initial_epsilon: float = 1.0,
                epsilon_decay: float = 1e-3,
                final_epsilon: float = 0.001,
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
        self.q_net = QNetwork(n_observations, hidden_dim, num_of_action, learning_rate,dropout).to(device)
        self.q_target = QNetwork(n_observations, hidden_dim, num_of_action, learning_rate,dropout).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.batch_size = batch_size
        self.steps_done = 0
        self.num_of_action = num_of_action
        self.tau = tau
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate,amsgrad=True)
        self.loss_fn = torch.nn.MSELoss()
        self.episode_durations = []

        self.update_target_networks()  # initialize target networks

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        pass
        # ====================================== #

        super(DQN, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
            
        )
        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()

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
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.num_of_action - 1)
        else:
            # state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # q_values = self.q_net(state_tensor)
                q_values = self.q_net(state)
            action_idx = q_values.argmax().item()
        
        # Map discrete action index to continuous action value
        scaled_action = self.scale_action(action_idx)
        # min_action, max_action = self.action_range
        # scaled_action = min_action + (action_idx / (self.num_of_action - 1)) * (max_action - min_action)

        # Add optional noise for exploration (useful for evaluation mode)
        # if noise > 0.0:
        #     scaled_action += np.random.normal(0, noise)
        #     scaled_action = np.clip(scaled_action, min_action, max_action)

        return scaled_action, action_idx
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
        if len(self.memory) < batch_size:
            return None
        # return self.memory.sample()
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*self.memory.sample())
        return (
            torch.stack([torch.tensor(s, dtype=torch.float32) for s in state_batch]).to(self.device),
            torch.tensor(action_batch, dtype=torch.int64, device=self.device),
            torch.tensor(reward_batch, dtype=torch.float32, device=self.device),
            torch.stack([torch.tensor(s, dtype=torch.float32) for s in next_state_batch]).to(self.device),
            torch.tensor(done_batch, dtype=torch.bool, device=self.device),
        )
        # ====================================== #
        
        # Sample a batch from memory
        # ========= put your code here ========= #
        # ====================================== #

    # def calculate_loss(self, states, actions, rewards, next_states, dones):
    def calculate_loss(self, state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask):
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

        # Gradient clipping for critic

        # Update Actor

        # Gradient clipping for actor
        states = state_batch
        actions = action_batch
        rewards = reward_batch
        next_states = non_final_next_states
        dones = non_final_mask.float()

        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # print(f'current_q = {current_q}')
        with torch.no_grad():
            next_q = self.q_target(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.discount_factor * next_q
        # print(f'next_q = {next_q}')
        loss = self.loss_fn(current_q, target_q)
        return loss
        # ====================================== #

    def update_policy(self):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        if len(self.memory) < self.batch_size:
            return
        sample = self.generate_sample(self.batch_size)
        if sample is None:
            return
        # print(sample)
        state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask = sample
        # ========= put your code here ========= #
        

        # Compute loss
        loss = self.calculate_loss(state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask)
        # print(f'loss = {loss}')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
        # ====================================== #


    def update_target_networks(self):
        """
        Perform soft update of target networks using Polyak averaging.

        Args:
            tau (float, optional): Update rate. Defaults to self.tau.
        """
        # ========= put your code here ========= #
        # tau = self.tau if tau is None else tau
        for target_param, param in zip(self.q_target.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        # ====================================== #

    def learn(self,env):
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
        done = False
        timestep = 0
        total_reward = 0.0
        total_loss = 0.0
        # ====================================== #
        while not done:
            # Predict action from the policy network
            # ========= put your code here ========= #
            state = obs['policy'].flatten()
            action,action_idx = self.select_action(state)
            # ====================================== #

            # Execute action in the environment and observe next state and reward
            # ========= put your code here ========= #
            action_tensor = torch.tensor([[action]], dtype=torch.float32).to(self.device)
            next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
            done = terminated or truncated
            total_reward += reward.item()
            
            next_state = next_obs['policy'].flatten()
            # state = obs['policy'].flatten().to(self.device)
            # next_state = next_obs['policy'].flatten().to(self.device)
            # ====================================== #

            # Store the transition in memory
            # ========= put your code here ========= #
            self.memory.add(state, action_idx, reward, next_state, done)
            # ====================================== #

            # Update state
            # Perform one step of the optimization (on the policy network)
            loss = self.update_policy()
            if loss is not None:
                total_loss += loss
            # Soft update of the target network's weights
            self.update_target_networks()
            obs = next_obs
            # print(obs)
            timestep += 1
            if done:
                self.plot_durations(timestep)
                break
        self.decay_epsilon()
        return total_reward, total_loss,timestep

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
        Save the Q-network weights to a file.
        
        Args:
            path (str): Path to save the model weights.
        """
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        torch.save(self.q_net.state_dict(), full_path)
        print(f"Model saved to {full_path}")

    def load_model(self, path, filename):
        """
        Load the Q-network weights from a file.
        
        Args:
            path (str): Path to the saved model weights.
        """
        full_path = os.path.join(path, filename)
        self.q_net.load_state_dict(torch.load(full_path, map_location=self.device))
        self.q_target.load_state_dict(self.q_net.state_dict())  # Sync target with policy
        print(f"Model loaded from {full_path}")
