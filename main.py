import os
import pandas as pd
import numpy as np
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns


from sim import Simulation
from log import ml, el, rl, gl
from config import log_path, size, markers, colors, default_marker, default_color
from env import Grid


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

    surf = sim.grid.surface

    resource_point_list = sim.grid.get_xy_res_pnt()
    resource_points_array = np.array(resource_point_list)

    # Base map for resources and surface
    def plot_base_map():
        plt.imshow(surf, cmap='terrain', extent=[0.0, float(size), 0.0, float(size)], origin='lower')
        plt.colorbar(label='Elevation')
        plt.scatter(resource_points_array[:, 0], resource_points_array[:, 1], c='red', label='Resource Points', marker='x')

    # Plotting encounters and creating heatmaps for each type
    grouped = enc_df.groupby('Interaction Type')
    for name, group in grouped:
        plt.figure()
        plot_base_map()
        encounter_points = group[['New X', 'New Y']].dropna().to_numpy()
        marker = markers.get(name, default_marker)
        color = colors.get(name, default_color)
        plt.scatter(encounter_points[:, 0], encounter_points[:, 1], marker=marker, color=color,
                    label=f'{name} Encounters', alpha=0.6)
        plt.title(f'Elevation Surface with Resource and {name} Points')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig(os.path.join(new_log_path, f"{name}_encounters.png"))
        plt.close()

        # Generate heatmap for the current interaction type
        heatmap_data = group.pivot_table(index='New Y', columns='New X', values='Epoch', aggfunc='count', fill_value=0)
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, cmap='viridis')
        plt.title(f'Heatmap of {name} Encounters')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        heatmap_filename = os.path.join(new_log_path, f"{name}_heatmap.png")
        plt.savefig(heatmap_filename)
        plt.close()

    plt.figure()
    plot_base_map()
    plt.title('Elevation Surface with Resource Points')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(os.path.join(new_log_path, "elevation_surface_with_resources.png"))
    plt.close()


    return metrics_df, move_df, enc_df, res_df, grp_df

if __name__ == '__main__':
    metrics_df, move_df, enc_df, res_df, grp_df = main()