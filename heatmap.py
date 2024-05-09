import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from config import new_log_path

# Load your data
enc = 'enc_log.csv'
path = os.path.join(new_log_path, enc)
data = pd.read_csv(path)

# Binning the data to prepare for heatmap
data['x_bin'] = pd.cut(data['New X'], bins=20, labels=False)  # Adjust 'bins' as needed
data['y_bin'] = pd.cut(data['New Y'], bins=20, labels=False)  # Adjust 'bins' as needed

# Creating a pivot table for the heatmap
heatmap_data = data.pivot_table(index='y_bin', columns='x_bin', values='New Z', aggfunc='count', fill_value=0)


plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=False, cmap='viridis')  # 'cmap' adjusts the color map
plt.title('Heatmap of Encounter Density')
plt.xlabel('X Coordinate Bin')
plt.ylabel('Y Coordinate Bin')
plt.show()

# Get unique interaction types
interaction_types = data['Interaction Type'].unique()

for interaction in interaction_types:
    # Filter data for the current interaction type
    subset = data[data['Interaction Type'] == interaction]

    # Create pivot table for heatmap
    heatmap_data = subset.pivot_table(index='y_bin', columns='x_bin', values='New Z', aggfunc='count', fill_value=0)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=False, cmap='viridis')  # Turn off annotations if the data is dense
    plt.title(f'Heatmap of {interaction} Encounter Density')
    plt.xlabel('X Coordinate Bin')
    plt.ylabel('Y Coordinate Bin')

    # Save the heatmap to the log path
    file_name = f"{interaction}_encounter_heatmap.png"
    file_path = os.path.join(new_log_path, file_name)
    plt.savefig(file_path)
    plt.show()
    plt.close()  # Close the plot to free up memory

