from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm
import matplotlib
import matplotlib.pyplot as plt
import torch

class Linear_QN(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            learning_rate: float = 0.01,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 0.999,
            final_epsilon: float = 0.001,
            discount_factor: float = 0.95,
    ) -> None:
        self.episode_durations = []
        """
        Initialize the CartPole Agent.

        Args:
            learning_rate (float): The learning rate for updating Q-values.
            initial_epsilon (float): The initial exploration rate.
            epsilon_decay (float): The rate at which epsilon decays over time.
            final_epsilon (float): The final exploration rate.
            discount_factor (float, optional): The discount factor for future rewards. Defaults to 0.95.
        """        
        
        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
        
    def update(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        next_action: int,
        terminated: bool
    ):
        """
        Updates the weight vector using the Temporal Difference (TD) error 
        in Q-learning with linear function approximation.

        Args:
            obs (dict): The current state observation, containing feature representations.
            action (int): The action taken in the current state.
            reward (float): The reward received for taking the action.
            next_obs (dict): The next state observation.
            next_action (int): The action taken in the next state (used in SARSA).
            terminated (bool): Whether the episode has ended.

        """
        # ========= put your code here ========= #
        obs_vec = obs["policy"].flatten()
        next_obs_vec = next_obs["policy"].flatten()

        q_next = 0.0 if terminated else self.q(next_obs_vec).max()
        q_target = reward + self.discount_factor * q_next
        q_current = self.q(obs_vec, action)
        td_error = q_target - q_current
        # Update weights
        td_error = td_error.item()  # Convert single-element tensor to float
        obs_vec = obs_vec.detach().cpu().numpy()
        # print(type(obs),type(self.lr),type(td_error),type(obs_vec))
        self.w[:, action] += self.lr * td_error * obs_vec

        # Log error
        self.training_error.append(td_error**2)
        # ====================================== #

    def select_action(self, state):
        """
        Select an action based on an epsilon-greedy policy.
        
        Args:
            state (Tensor): The current state of the environment.
        
        Returns:
            Tensor: The selected action.
        """
        # ========= put your code here ========= #
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_of_action)
        q_vals = self.q(state)
        # print(q_vals)
        return int(np.argmax(q_vals))
        # ====================================== #

    def learn(self, env):
        """
        Train the agent on a single step.

        Args:
            env: The environment in which the agent interacts.
            max_steps (int): Maximum number of steps per episode.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        obs,_ = env.reset()
        total_reward = 0.0
        done = False
        timestep = 0

        while not done:
            state = obs['policy']
            action = self.select_action(state)
            scaled_action = self.scale_action(action)
            action_tensor = torch.tensor([[scaled_action]], dtype=torch.float32)
            next_obs, reward, terminated, truncated, _  = env.step(action_tensor)
            next_state = next_obs['policy']
            next_action = self.select_action(next_state)
            self.update(obs, action, reward.item(), next_obs, next_action, done)
            obs = next_obs
            total_reward += reward.item()
            done = terminated or truncated


            timestep += 1
            if done:
                self.plot_durations(timestep)
                break
        self.decay_epsilon()
        return total_reward,timestep
        # ====================================== #

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




    