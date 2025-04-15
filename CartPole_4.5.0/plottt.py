import json
import numpy as np
import matplotlib.pyplot as plt

# Load the Q-values from the file
file_path = "./q_value/Stabilize/Q_Learning/Finalbase/Q_Learning_4900_11_12.0_5_11.json"  # Update this with the correct file path
with open(file_path, "r") as f:
    data = json.load(f)

q_values = data["q_values"]

# Extract the best Q-value for each (cart position, pole position) pair
q_table = {}
for key, values in q_values.items():
    cart_pos, pole_pos, _, _ = eval(key)  # Extract cart position and pole position
    best_q_value = max(values)  # Get the best Q-value for the state
    if (cart_pos, pole_pos) not in q_table:
        q_table[(cart_pos, pole_pos)] = best_q_value
    else:
        q_table[(cart_pos, pole_pos)] = max(q_table[(cart_pos, pole_pos)], best_q_value)

# Convert the dictionary to a 2D array for plotting
cart_positions = sorted(set(k[0] for k in q_table.keys()))
pole_positions = sorted(set(k[1] for k in q_table.keys()))

q_matrix = np.zeros((len(cart_positions), len(pole_positions)))

for i, cart in enumerate(cart_positions):
    for j, pole in enumerate(pole_positions):
        q_matrix[i, j] = q_table.get((cart, pole), np.nan)  # Use NaN for missing values

# Plot the heatmap
plt.figure(figsize=(8, 6))
plt.imshow(q_matrix, cmap="viridis", origin="lower", aspect="auto")
plt.colorbar(label="Best Q-Value")
plt.xticks(ticks=np.arange(len(pole_positions)), labels=pole_positions)
plt.yticks(ticks=np.arange(len(cart_positions)), labels=cart_positions)
plt.xlabel("Pole Position")
plt.ylabel("Cart Position")
plt.title("Best Q-Values for Cart Position and Pole Position")
plt.show()
