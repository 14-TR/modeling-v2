# ==================================================================
import random
from surface_noise import generate_noise
from perlin import Perlin
from config import vi, vj, z


class DayTracker:
    current_day = 1

    @classmethod
    def increment_day(cls):
        cls.current_day += 1

    @classmethod
    def get_current_day(cls):
        return cls.current_day

    @classmethod
    def reset(cls):
        cls.current_day = 1


# ------------------------------------------------------------------


class Epoch:
    epoch = 1

    @classmethod
    def increment_sim(cls):
        cls.epoch += 1

    @classmethod
    def get_current_epoch(cls):
        return cls.epoch

    @classmethod
    def reset(cls):
        cls.epoch = 1


# ------------------------------------------------------------------


class Grid:
    def __init__(self, grid_size):
        self.width = grid_size[0]
        self.height = grid_size[1]
        self.ents = []
        self.rmv_ents = 0
        self.occupied_positions = set()
        self.surface = generate_noise(grid_size,grid_size,vi,vj,z)
        self.resource_points = None

    def gen_res_pnt(self, num_points):
        points = set()
        while len(points) < num_points:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            points.add((x, y))
        self.resource_points = points
        return points

    def get_xy_res_pnt(self):
        return [(x, y) for x, y in self.resource_points]

    def get_nearest_res_pnt(self, entity_x, entity_y):
        # Initialize minimum distance and closest resource point variables
        min_distance = float('inf')
        closest_resource_point = None

        # Iterate through each resource point to find the closest one
        for resource in self.resource_points:
            # Calculate the Euclidean distance from the entity to this resource point
            distance = self.calc_dist(entity_x, entity_y, resource[ 'x' ], resource[ 'y' ])

            # If this resource point is closer than the previous closest, update min_distance and closest_resource_point
            if distance < min_distance:
                min_distance = distance
                closest_resource_point = resource

        return closest_resource_point[ 'x' ], closest_resource_point[ 'y' ]

    def get_distance_to_nearest_res_pnt(self, entity_x, entity_y):
        # Initialize minimum distance
        min_distance = float('inf')

        # Iterate through each resource point to find the closest one
        for resource in self.resource_points:
            # Calculate the Euclidean distance from the entity to this resource point
            distance = self.calc_dist(entity_x, entity_y, resource[0], resource[1])

            # If this resource point is closer than the previous closest, update min_distance
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def get_nearest_prey(self, entity):
        # Initialize minimum distance and closest prey variables
        min_distance = float('inf')
        closest_prey = None

        # Define the boundaries of the 5x5 grid around the entity
        min_x, max_x = entity.loc['x'] - 2, entity.loc['x'] + 2
        min_y, max_y = entity.loc['y'] - 2, entity.loc['y'] + 2

        # Iterate through each prey to find the closest one within the 5x5 grid
        for prey in self.ents:
            if not prey.is_z and min_x <= prey.loc['x'] <= max_x and min_y <= prey.loc['y'] <= max_y:
                # Calculate the Euclidean distance from the entity to this prey
                distance = self.calc_dist(entity.loc['x'], entity.loc['y'], prey.loc['x'], prey.loc['y'])

                # If this prey is closer than the previous closest, update min_distance and closest_prey
                if distance < min_distance:
                    min_distance = distance
                    closest_prey = prey

        # Return the x and y coordinates of the closest prey
        return (closest_prey.loc['x'], closest_prey.loc['y']) if closest_prey else None

    @staticmethod
    def calc_dist(x1, y1, x2, y2):
        # Calculate the Euclidean distance between two points
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def add_ent(self, ent):
        while (ent.x, ent.y) in self.occupied_positions:  # Check if position is occupied
            # Generate new positions until an unoccupied one is found
            ent.x = random.randint(0, self.width - 1)
            ent.y = random.randint(0, self.height - 1)
        self.occupied_positions.add((ent.x, ent.y))  # Mark the new position as occupied
        self.ents.append(ent)

    # def move_being(self, entity):
    #     h_move = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    #     z_move = [-2, -1, 0, 1, 2]
    #     if not entity.is_zombie:
    #         dx = random.choice(z_move)
    #         dy = random.choice(z_move)
    #     else:
    #         dx = random.choice(h_move)
    #         dy = random.choice(h_move)
    #
    #     # Store the current position
    #     # current_x, current_y, current_z = entity.x, entity.y, entity.z
    #     current_x, current_y = entity.x, entity.y
    #
    #     # Being attempts to move (the move method will handle resource/lifespan checks)
    #     entity.move(dx, dy, self, self.resource_points)
    #
    #     # If the move was successful (the being's position changed), update occupied positions
    #     if (current_x, current_y) != (entity.x, entity.y):
    #         # Remove the old position if it exists in the set
    #         if (current_x, current_y) in self.occupied_positions:
    #             self.occupied_positions.remove((current_x, current_y))
    #
    #         # Add the new position
    #         self.occupied_positions.add((entity.x, entity.y))

    def simulate_day(self):
        for ent in self.ents:
            ent.move(self)
            ent.update_status()

        self.remove_inactive_ents()

    def remove_inactive_ents(self):
        self.ents = [ent for ent in self.ents if ent.is_active]
        # add to the number of beings removed the number self.beings list gains
        self.occupied_positions = {(ent.loc['x'], ent.loc['y']) for ent in self.ents}

    # def count_humans_and_zombies(self):
    #     humans = sum(1 for being in self.beings if not being.is_zombie and being.is_active)
    #     zombies = sum(1 for being in self.beings if being.is_zombie and being.is_active)
    #     return humans, zombies

    # append the surface generated in surface_noise.py to the grid
    def append_surface(self, surface):
        self.surface = surface

    def get_elev_at(self, x, y):
        if self.surface is None:
            return 0
        else:
            return self.surface[x, y]