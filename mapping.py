import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# def generate_scatter_plot(image_path, locations, event_type, save_dir):
#     fig, ax = plt.subplots()
#
#     # Load the image into a numpy array
#     grid = mpimg.imread(image_path)
#
#     ax.imshow(grid, cmap='terrain', origin='lower')
#
#     # ... rest of your code ...
#
#     # Plot the grid surface
#     x_coords = [loc[0] for loc in locations]
#     y_coords = [loc[1] for loc in locations]
#
#     ax.imshow(grid, cmap='gray', origin='lower')
#
#     # Plot the scatter plot on top of the grid surface
#     ax.scatter(x_coords, y_coords, alpha=0.5)
#
#     # Plot the encounter locations on top of the grid surface
#     ax.set_xlabel('X')
#     ax.scatter(x_coords, y_coords, c='r', alpha=0.5)
#     ax.set_ylabel('Y')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_title(f'Interaction Type: {event_type}')
#
#     ax.set_title(f'Interaction Type: {event_type}')
#     plt.savefig(os.path.join(save_dir, f'{event_type}_scatter_plot.png'))
#     plt.show()

def plot_elevation_surface(grid):
    fig, ax = plt.subplots()

    # Display the grid surface as an image
    ax.imshow(grid, cmap='terrain', origin='lower')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Elevation Surface')

    plt.show()

