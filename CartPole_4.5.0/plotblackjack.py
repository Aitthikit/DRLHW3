import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from collections import defaultdict
from matplotlib.patches import Patch
from collections import defaultdict


# Load JSON data
file_path = "./q_value/Stabilize/Q_Learning/Finalbase/Q_Learning_4900_11_12.0_5_11.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Extract Q-values from the data
q_values = {eval(k): v for k, v in data["q_values"].items()}
print(q_values.values())

# # Fix cart position and velocity
# fixed_cart_position = 0.0
# fixed_cart_velocity = 0.0

# # Define range for pole angle and angular velocity
# pole_angle_range = np.linspace(-0.2, 0.2, 20)  # Radians
# pole_velocity_range = np.linspace(-1.0, 1.0, 20)  # Angular velocity

# # Create meshgrid for visualization
# pole_angle_grid, pole_velocity_grid = np.meshgrid(pole_angle_range, pole_velocity_range)

# # Initialize grids
# value_grid = np.zeros_like(pole_angle_grid)
# policy_grid = np.zeros_like(pole_angle_grid, dtype=int)
# # Convert Q-values to state values and policies
# state_value = defaultdict(float)
# policy = defaultdict(int)
# for obs, action_values in q_values.items():
#     state_value[obs] = float(np.max(action_values))
#     policy[obs] = int(np.argmax(action_values))

# # Define grid space for pole angle and pole angular velocity
# pole_angle_range = np.linspace(-0.2, 0.2, 20)  # Radians
# pole_velocity_range = np.linspace(-1.0, 1.0, 20)  # Angular velocity
# pole_angle_grid, pole_velocity_grid = np.meshgrid(pole_angle_range, pole_velocity_range)

# # Create value and policy grids
# value = np.apply_along_axis(
#     lambda obs: state_value.get((fixed_cart_position, fixed_cart_velocity, obs[0], obs[1]), 0),
#     axis=2,
#     arr=np.dstack([pole_angle_grid, pole_velocity_grid]),
# )
# value_grid = pole_angle_grid, pole_velocity_grid, value

# policy_grid = np.apply_along_axis(
#     lambda obs: policy.get((fixed_cart_position, fixed_cart_velocity, obs[0], obs[1]), -1),
#     axis=2,
#     arr=np.dstack([pole_angle_grid, pole_velocity_grid]),
# )

# # Plot function
# def create_cartpole_plots(value_grid, policy_grid, title):
#     pole_angle_grid, pole_velocity_grid, value = value_grid
#     fig = plt.figure(figsize=plt.figaspect(0.4))
#     fig.suptitle(title, fontsize=16)

#     # 3D Plot of State Values
#     ax1 = fig.add_subplot(1, 2, 1, projection="3d")
#     ax1.plot_surface(
#         pole_angle_grid, pole_velocity_grid, value, rstride=1, cstride=1, cmap="viridis", edgecolor="none"
#     )
#     ax1.set_title(f"State values: {title}")
#     ax1.set_xlabel("Pole Angle (rad)")
#     ax1.set_ylabel("Pole Angular Velocity")
#     ax1.set_zlabel("Value", fontsize=14)
#     ax1.view_init(20, 220)

#     # Policy Heatmap
#     fig.add_subplot(1, 2, 2)
#     ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
#     ax2.set_title(f"Policy: {title}")
#     ax2.set_xlabel("Pole Angle (rad)")
#     ax2.set_ylabel("Pole Angular Velocity")
#     ax2.set_xticks(np.linspace(0, len(pole_angle_range) - 1, 5))
#     ax2.set_xticklabels(np.round(np.linspace(-0.2, 0.2, 5), 2))
#     ax2.set_yticks(np.linspace(0, len(pole_velocity_range) - 1, 5))
#     ax2.set_yticklabels(np.round(np.linspace(-1.0, 1.0, 5), 2))

#     # Add legend for policy actions
#     legend_elements = [
#         Patch(facecolor="lightgreen", edgecolor="black", label="Move Left"),
#         Patch(facecolor="grey", edgecolor="black", label="Move Right"),
#     ]
#     ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))

#     return fig

# # Create and show plots
# fig = create_cartpole_plots(value_grid, policy_grid, title="CartPole Q-Learning Results")
# plt.show()
