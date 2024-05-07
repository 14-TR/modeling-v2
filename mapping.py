import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from log import EncLog
# from config import W, H
from env import Grid

def plot_surface(surface):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(0, surface.shape[0])
    Y = np.arange(0, surface.shape[1])
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, surface, cmap='terrain')
    plt.show()

def collect_enc_locations(event_type):
    enc_log_instance = EncLog()  # Assuming you have a Log instance
    event_records = enc_log_instance.get_records_by_type(event_type)
    locations = [(record.x, record.y, record.z) for record in event_records]  # Collecting locations from records
    return locations

def generate_scatter_plot(locations, event_type):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_coords = [loc[0] for loc in locations]
    y_coords = [loc[1] for loc in locations]
    z_coords = [loc[2] for loc in locations]
    ax.scatter(x_coords, y_coords, z_coords)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Interaction Type: {event_type}')
    plt.show()