import os

import numpy as np
from matplotlib import pyplot as plt
from config import grid_size


class SurfaceMapper:
    def __init__(self, elevation_data, path, grid_size=grid_size):
        self.elevation_data = elevation_data
        self.path = path
        self.grid_size = grid_size

    def plot_surface(self):
        # Create a meshgrid for the x and y coordinates
        x = np.linspace(0, self.grid_size[0], num=self.elevation_data.shape[1])
        y = np.linspace(0, self.grid_size[1], num=self.elevation_data.shape[0])
        x, y = np.meshgrid(x, y)

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        ax.plot_surface(x, y, self.elevation_data, cmap='terrain')

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Elevation')

        # Show the plot
        plt.show()
        plt.savefig(os.path.join(self.path, "elevation_surface.png"))