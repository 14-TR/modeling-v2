import os
import pandas as pd
import numpy as np
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from sim import Simulation
from log import ml, el, rl, gl
from config import log_path
from env import Grid
from mapping import plot_surface, collect_enc_locations, generate_scatter_plot

import matplotlib.pyplot as plt

def main():
    # Initialize the simulation
    sim = Simulation()

    # Generate a unique folder name using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_log_path = os.path.join(log_path, timestamp)

    # Create the new folder
    os.makedirs(new_log_path, exist_ok=True)

    # Initialize an empty DataFrame for the encounter logs
    enc_df = pd.DataFrame(columns=["Epoch", "Day", "Entity 1", "Entity 2", "Interaction Type", "New X", "New Y", "New Z"])

    # Run the simulation and get the metrics dictionary and encounter logs
    metrics_list, encounter_logs = sim.run()

    # Convert the metrics dictionary to a DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # Append the encounter logs for each epoch to the DataFrame
    for logs in encounter_logs:
        enc_df = enc_df._append(pd.DataFrame([str(record).split(',') for record in logs],
                                            columns=["Epoch", "Day", "Entity 1", "Entity 2", "Interaction Type",
                                                     "New X", "New Y", "New Z"]))

    # Create dataframes for each log and write them to CSV files in the new folder
    move_df = pd.DataFrame([str(record).split(',') for record in ml.logs],
                           columns=["Epoch", "Day", "Entity", "Old X", "Old Y", "Old Z", "New X", "New Y", "New Z"])
    move_df.to_csv(os.path.join(new_log_path, "move_log.csv"), index=False)

    res_df = pd.DataFrame([str(record).split(',') for record in rl.logs],
                          columns=["Epoch", "Day", "Entity", "Resource Change", "Current Resources", "Reason"])
    res_df.to_csv(os.path.join(new_log_path, "res_log.csv"), index=False)

    grp_df = pd.DataFrame([str(record).split(',') for record in gl.logs],
                          columns=["Epoch", "Day", "Group", "Entity", "Action", "Reason"])
    grp_df.to_csv(os.path.join(new_log_path, "grp_log.csv"), index=False)

    # Write the metrics DataFrame to a CSV file
    metrics_df.to_csv(os.path.join(new_log_path, "metrics_log.csv"), index=False)

    # Write the encounter logs DataFrame to a CSV file
    enc_df.to_csv(os.path.join(new_log_path, "enc_log.csv"), index=False)

    # Convert 'New X', 'New Y', 'New Z' columns to numeric data types
    enc_df['New X'] = pd.to_numeric(enc_df['New X'], errors='coerce')
    enc_df['New Y'] = pd.to_numeric(enc_df['New Y'], errors='coerce')
    enc_df['New Z'] = pd.to_numeric(enc_df['New Z'], errors='coerce')

    # Group the DataFrame by the 'Interaction Type' column
    grouped = enc_df.groupby('Interaction Type')

    # Create a scatter plot for each group
    for name, group in grouped:
        x_coords = group['New X'].values
        y_coords = group['New Y'].values
        z_coords = group['New Z'].values

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_coords, y_coords, z_coords)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Interaction Type: {name}')

        plt.show()

    # Plot the surface
    grid = Grid((100, 100))  # Replace with your actual grid
    plot_surface(grid.surface)

    return metrics_df, move_df, enc_df, res_df, grp_df

if __name__ == '__main__':
    metrics_df, move_df, enc_df, res_df, grp_df = main()