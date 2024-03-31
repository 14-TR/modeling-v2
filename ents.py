import random

from env import Grid
from util import id_generator
from log import ml, gl, el, rl, MoveRecord, ResRecord
from config import start_res, start_ttd, grid_size, max_res_gain


class Entity:
    grid = Grid(*grid_size)

    def __init__(self, grid=grid):
        self.id = id_generator.gen_id()  # Generate a unique ID for the entity
        self.loc = {'x': 0, 'y': 0, 'z': 0}  # Default location
        self.att = {'res': 0, 'ttd': 0}  # Default attributes
        self.is_z = False  # Default zombie flag
        self.enc = {'luv': 0, 'war': 0, 'rob': 0, 'esc': 0, 'win': 0}  # Default encounters
        self.xp = {'luv_xp': 0, 'war_xp': 0, 'esc_xp': 0, 'win_xp': 0}
        self.node = {'human': False, 'zombie': False}  # Default node flags
        self.grp = {}  # Default groups, group ids held as keys
        self.net = {'friend': {}, 'foe': {}} # Default network connections, entity ids held as keys
        self.grid = grid

    def is_adjacent(self, other):
        dx = abs(self.loc['x'] - other.loc['x'])
        dy = abs(self.loc['y'] - other.loc['y'])
        return dx in [0, 1] and dy in [0, 1]

    def interact(self, other):
        from events import interact  # Import the interact function from events.py
        interact(self, other)  # Call the interact function with self and other as arguments

    def update_status(self):
        from events import update_status
        update_status(self)

    def calculate_elevation_effect(self, dx, dy):
        # Calculate the elevation difference
        current_elevation = self.grid.get_elev_at(self.loc['x'], self.loc['y'])
        target_elevation = self.grid.get_elev_at(self.loc['x'] + dx, self.loc['y'] + dy)
        elevation_diff = target_elevation - current_elevation

        return elevation_diff


class Human(Entity):

    def __init__(self, res=start_res):
        super().__init__()
        self.att['res'] = res

    def move(self):
        old_loc = self.loc.copy()
        if self.grp and self.prob_move_grp():
            self.move_grp()
        elif self.prob_move_res():
            self.move_res()
        else:
            self.move_rand()

        mr = MoveRecord(self, old_loc, self.loc)
        ml.log(mr)

        self.replenish_resources()
        self.distribute_resources()

        self.att['res'] -= .5

        rr = ResRecord(self, -0.5, 'move')
        rl.log(rr)

    def prob_move_res(self):
        return random.random() < (1 - self.att['res'] / 10)

    def prob_move_grp(self):
        collective_res = sum(member.att['res'] for member in self.grp.values()) / len(self.grp)
        return collective_res < 5

    def move_rand(self):
        # Random movement logic, modifying x and y coordinates
        step_range = (-4, -3, -2, -1, 1, 2, 3, 4)  # Define the range of steps (inclusive). Adjust as needed.
        dx = random.randint(*step_range)  # Step size for x direction
        dy = random.randint(*step_range)  # Step size for y direction

        elevation_diff = self.calculate_elevation_effect(dx, dy)

        # Subtract resources proportional to the elevation difference
        self.att[ 'res' ] -= abs(elevation_diff)

        # Update location
        self.loc[ 'x' ] += dx
        self.loc[ 'y' ] += dy

    def move_res(self):
        # Assuming the Grid class has a method get_nearest_resource_point() that returns the closest resource point's (x, y) location
        nearest_res_x, nearest_res_y = self.grid.get_nearest_res_pnt(self.loc[ 'x' ], self.loc[ 'y' ])

        # Calculate direction to move towards the resource point. This is a simple version that moves one step towards the resource point.
        dx = 1 if nearest_res_x > self.loc[ 'x' ] else -1 if nearest_res_x < self.loc[ 'x' ] else 0
        dy = 1 if nearest_res_y > self.loc[ 'y' ] else -1 if nearest_res_y < self.loc[ 'y' ] else 0

        elevation_diff = self.calculate_elevation_effect(dx, dy)

        self.att[ 'res' ] -= abs(elevation_diff)

        # Update location towards the resource point
        self.loc[ 'x' ] += dx
        self.loc[ 'y' ] += dy

        print(f"Moving towards resource: New location ({self.loc[ 'x' ]}, {self.loc[ 'y' ]})")

    def move_grp(self):
        # Calculate the average position of the group
        avg_x = sum(member.loc[ 'x' ] for member in self.grp.values()) / len(self.grp)
        avg_y = sum(member.loc[ 'y' ] for member in self.grp.values()) / len(self.grp)

        # Determine the direction to move towards the group's average position
        dx = 1 if avg_x > self.loc[ 'x' ] else -1 if avg_x < self.loc[ 'x' ] else 0
        dy = 1 if avg_y > self.loc[ 'y' ] else -1 if avg_y < self.loc[ 'y' ] else 0

        elevation_diff = self.calculate_elevation_effect(dx, dy)

        self.att[ 'res' ] -= abs(elevation_diff)

        # Update location towards the group's average position
        self.loc[ 'x' ] += dx
        self.loc[ 'y' ] += dy

        print(f"Moving with group: New location ({self.loc[ 'x' ]}, {self.loc[ 'y' ]})")

    def replenish_resources(self):
        # Check if the current location is a resource point
        if (self.loc['x'], self.loc['y']) in self.grid.resource_points:
            # Increase resources by a random amount up to max_res_gain
            self.att['res'] += random.randint(max_res_gain*.5, max_res_gain)

            # Remove the resource point from the grid
            self.grid.resource_points.remove((self.loc['x'], self.loc['y']))

    def distribute_resources(self):
        # Find adjacent group members
        adjacent_group_members = [member for member in self.grp.values() if self.is_adjacent(member)]

        # Calculate total resources of self and adjacent group members
        total_resources = self.att['res'] + sum(member.att['res'] for member in adjacent_group_members)

        # Distribute resources evenly among self and adjacent group members
        if adjacent_group_members:
            distributed_resource = total_resources / (len(adjacent_group_members) + 1)
            self.att['res'] = round(distributed_resource)
            for member in adjacent_group_members:
                member.att['res'] = distributed_resource


class Zombie(Entity):

    def __init__(self, ttd=start_ttd):
        super().__init__()
        self.att['ttd'] = ttd


    def move(self):
        og_loc = self.loc.copy()

        if self.grp and self.prob_move_grp():
            self.move_grp()
        elif self.prob_move_prey():
            self.move_prey()
        else:
            self.move_rand()

        rec = MoveRecord(self, og_loc, self.loc)
        ml.log(rec)

        self.att['ttd'] -= 1.5

    def prob_move_grp(self):
        collective_res = sum(member.res for member in self.grp.values()) / len(self.grp)
        return collective_res < 5

    def prob_move_prey(self):
        return random.random() < (1 - self.att['ttd'] / 10)

    def move_rand(self):
        # Random movement logic, modifying x and y coordinates
        step_range = (-2, -1, 1, 2)  # Define the range of steps (inclusive). Adjust as needed.
        dx = random.randint(*step_range)  # Step size for x direction
        dy = random.randint(*step_range)  # Step size for y direction

        elevation_diff = self.calculate_elevation_effect(dx, dy)

        # If moving uphill, reduce the step size
        if elevation_diff > 0:
            dx = round(dx / 2)
            dy = round(dy / 2)

        # Update location
        self.loc[ 'x' ] += dx
        self.loc[ 'y' ] += dy


    def move_grp(self):
        # Calculate the average position of the group
        avg_x = sum(member.loc[ 'x' ] for member in self.grp.values()) / len(self.grp)
        avg_y = sum(member.loc[ 'y' ] for member in self.grp.values()) / len(self.grp)

        # Determine the direction to move towards the group's average position
        dx = 1 if avg_x > self.loc[ 'x' ] else -1 if avg_x < self.loc[ 'x' ] else 0
        dy = 1 if avg_y > self.loc[ 'y' ] else -1 if avg_y < self.loc[ 'y' ] else 0

        # Update location towards the group's average position
        self.loc[ 'x' ] += dx
        self.loc[ 'y' ] += dy

        print(f"Moving with group: New location ({self.loc[ 'x' ]}, {self.loc[ 'y' ]})")

    def move_prey(self):
        nearest_prey_coords = self.grid.get_nearest_prey(self)

        if nearest_prey_coords is not None:
            nearest_prey_x, nearest_prey_y = nearest_prey_coords

            # Calculate direction to move towards the human. This is a simple version that moves one step towards the
            # human.
            dx = 1 if nearest_prey_x > self.loc['x'] else -1 if nearest_prey_x < self.loc['x'] else 0
            dy = 1 if nearest_prey_y > self.loc['y'] else -1 if nearest_prey_y < self.loc['y'] else 0

            elevation_diff = self.calculate_elevation_effect(dx, dy)

            # If moving uphill, reduce the step size
            if elevation_diff > 0:
                dx = round(dx / 2)
                dy = round(dy / 2)

            # Update location towards the human
            self.loc['x'] += dx
            self.loc['y'] += dy

            print(f"Moving towards prey: New location ({self.loc['x']}, {self.loc['y']})")
        else:
            # If there's no prey within the 5x5 grid, the zombie can move randomly or stay in place
            self.move_rand()


class Group:
    def __init__(self):
        self.id = id_generator.gen_id()
        self.members = []  # List of Entity objects

    def add_member(self, entity):
        self.members.append(entity)
        gl.log(self, entity, 'add')

    def remove_member(self, entity):
        for member in self.members:
            if member.is_z == entity.is_z and member.id == entity.id:
                gl.log(self, entity, 'remove')
                self.members.remove(member)
                break

    def interact(self, other):
        from events import interact  # Import the interact function from events.py

        if isinstance(other, Group):  # If the other is a group
            for member in self.members:
                for other_member in other.members:
                    interact(member, other_member)  # Call the interact function with each pair of entities
        else:  # If the other is an individual entity
            for member in self.members:
                interact(member, other)  # Call the interact function with each entity and the individual entity
