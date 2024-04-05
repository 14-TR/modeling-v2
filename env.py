# ==================================================================
import random
# from ents import entities

from config import vi, vj, z, w, h
from surface_noise import generate_noise


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
        self.surface = generate_noise(w, h, vi, vj, z)
        self.resource_points = set()

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
            distance = self.calc_dist(entity_x, entity_y, resource['x'], resource['y'])

            # If this resource point is closer than the previous closest, update min_distance and closest_resource_point
            if distance < min_distance:
                min_distance = distance
                closest_resource_point = resource

        if closest_resource_point is None:
            return (0, 0)  # Default value

        return closest_resource_point['x'], closest_resource_point['y']

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
        while (ent.loc['x'], ent.loc['y']) in self.occupied_positions:  # Check if position is occupied
            # Generate new positions until an unoccupied one is found
            ent.loc['x'] = random.randint(1, self.width - 1)
            ent.loc['y'] = random.randint(1, self.height - 1)
        self.occupied_positions.add((ent.loc['x'], ent.loc['y']))  # Add the new position to the set
        self.ents.append(ent)

    def remove_inactive_ents(self):
        for ent in self.ents:
            if not ent.is_active:
                if (ent.loc['x'], ent.loc['y']) in self.occupied_positions:
                    self.occupied_positions.remove((ent.loc['x'], ent.loc['y']))
                self.ents.remove(ent)
                self.rmv_ents += 1

    def simulate_day(self):
        for ent in self.ents:
            ent.move(self)
            ent.update_status()

        self.remove_inactive_ents()

        # Increment the day in the simulation
        DayTracker.increment_day()

    def get_elevation(self, x, y):
        return self.surface[x][y]

    # append the surface generated in surface_noise.py to the grid
    def append_surface(self, surface):
        self.surface = surface

    def is_in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    # def count_humans_and_zombies(self):
    #     humans = sum(1 for being in self.beings if not being.is_zombie and being.is_active)
    #     zombies = sum(1 for being in self.beings if being.is_zombie and being.is_active)
    #     return humans, zombies

