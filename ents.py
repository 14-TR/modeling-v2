import random

from config import start_res, start_ttd, grid_size, max_res_gain
from env import Grid
# from globals import global_entities
from log import ml, gl, rl, MoveRecord, ResRecord
from util import id_generator

entities = {}  # Dictionary to store all entities


class Entity:
    grid = Grid(grid_size=grid_size)

    def __init__(self, grid=grid, entity_type=''):
        self.id = id_generator.gen_id(entity_type)  # Generate a unique ID for the entity
        entities[self.id] = self  # Add the entity to the global entities dictionary
        # print(len(entities))
        self.loc = {'x': 0, 'y': 0, 'z': 0}  # Default location
        self.att = {'res': 0, 'ttd': 0}  # Default attributes
        self.is_z = False  # Default zombie flag
        self.is_active = True  # Default active flag
        self.enc = {'luv': 0, 'war': 0, 'rob': 0, 'esc': 0, 'win': 0, 'inf': 0}  # Default encounters
        self.xp = {'luv': 0, 'war': 0, 'rob': 0, 'esc': 0, 'win': 0}
        self.node = {'human': False, 'zombie': False}  # Default node flags
        self.grp = {}  # Default groups, group ids held as keys
        self.net = {'friend': {}, 'foe': {}}  # Default network connections, entity ids held as keys
        self.grid = grid
        # self.day = 0

    def is_adjacent(self, other):
        dx = abs(self.loc['x'] - other.loc['x'])
        dy = abs(self.loc['y'] - other.loc['y'])
        return dx in [0, 1] and dy in [0, 1]

    def interact(self, other):
        from events import interact  # Import the interact function from events.py
        interact(ent1=self, ent2=other)  # Call the interact function with self and other as arguments

    def update_status(self, simulation):
        from events import update_status
        update_status(self, simulation)
        if not self.is_active:
            # global_entities['removed'].append(self)
            self.id = self.id.split('_')[0] + '_X'

    def calculate_elevation_effect(self, dx, dy):
        # Calculate the elevation difference
        current_elevation = self.grid.get_elevation(self.loc['x'], self.loc['y'])
        target_elevation = self.grid.get_elevation(self.loc['x'] + dx, self.loc['y'] + dy)
        elevation_diff = target_elevation - current_elevation

        return elevation_diff


class Human(Entity):

    def __init__(self, res=start_res):
        super().__init__(entity_type='H')
        # super().__init__()
        # replace the self id for the entity related to this human in the entities dict
        self.att['res'] = res
        self.is_h = True
        self.is_z = False
        self.is_active = True
        # append the human to the entities dict
        entities[self.id] = self
        self.starved = False
        # global_entities['humans'].append(self)

    def __str__(self):
        return f"Human {self.id}"

    def move(self, simulation):
        if not self.is_z:
            old_loc = self.loc.copy()
            if self.grp and self.prob_move_grp():
                self.move_grp()
            elif self.prob_move_res():
                self.move_res()
            else:
                self.move_rand()

            ml.log(self, old_loc['x'], old_loc['y'], old_loc['z'], self.loc['x'], self.loc['y'], self.loc['z'])

            self.replenish_resources()
            self.distribute_resources()

            self.att['res'] -= .5
            # self.day += 1
            rl.log(entity=self, res_change=-.5, reason='move')

            # Check if the human has run out of resources

            # if self.att['res'] <= 0:
            # print(f"Human {self.id} has run out of resources.")
            # starve_cnt.append(self.id)
            # self.turn_into_zombie(on_turn_into_zombie_callback=simulation.handle_turn_into_zombie)
            # print(f"Starve count: {len(starve_cnt)}")

    def prob_move_res(self):
        return random.random() < (1 - self.att['res'] / 10)

    def prob_move_grp(self):
        # Calculate the total resources of all group members
        total_resources = 0
        total_members = 0
        for group_id in self.grp.keys():
            if group_id in entities:  # Check if the Group object exists
                group = entities[group_id]  # Get the Group object
                for member_id in group.members:
                    member = group.get_member_by_id(member_id)  # Get the Entity object
                    if member is not None:
                        total_resources += member.att['res']
                        total_members += 1

        # Calculate the average resources of the group members
        collective_res = total_resources / total_members if total_members > 0 else 0

        return collective_res < 5

    def move_rand(self):
        # Random movement logic, modifying x and y coordinates
        # step_range = (-4, -3, -2, -1, 1, 2, 3, 4)  # Define the range of steps (inclusive). Adjust as needed.
        dx = random.randint(-4, 4)  # Step size for x direction
        dy = random.randint(-4, 4)  # Step size for y direction

        if self.grid.is_in_bounds(self.loc['x'] + dx, self.loc['y'] + dy):
            elevation_diff = self.calculate_elevation_effect(dx, dy)

            # Subtract resources proportional to the elevation difference
            self.att['res'] -= abs(elevation_diff)

            # Update location
            self.loc['x'] += dx
            self.loc['y'] += dy

    def move_res(self):
        # Assuming the Grid class has a method get_nearest_resource_point() that returns the closest resource point's
        # (x, y) location
        nearest_res_x, nearest_res_y = self.grid.get_nearest_res_pnt(self.loc['x'], self.loc['y'])

        # Calculate direction to move towards the resource point. This is a simple version that moves one step
        # towards the resource point.
        dx = 1 if nearest_res_x > self.loc['x'] else -1 if nearest_res_x < self.loc['x'] else 0
        dy = 1 if nearest_res_y > self.loc['y'] else -1 if nearest_res_y < self.loc['y'] else 0

        elevation_diff = self.calculate_elevation_effect(dx, dy)

        self.att['res'] -= abs(elevation_diff)

        # Update location towards the resource point
        self.loc['x'] += dx
        self.loc['y'] += dy

        # print(f"Moving towards resource: New location ({self.loc[ 'x' ]}, {self.loc[ 'y' ]})")

    def move_grp(self):
        total_members = 0
        total_x = 0
        total_y = 0

        for group in self.grp.values():
            for member_id in group.members:
                member = group.get_member_by_id(member_id)  # Get the Entity object
                if member is not None:
                    total_x += member.loc['x']
                    total_y += member.loc['y']
                    total_members += 1

        if total_members > 0:
            avg_x = total_x / total_members
            avg_y = total_y / total_members

            # Determine the direction to move towards the group's average position
            dx = 1 if avg_x > self.loc['x'] else -1 if avg_x < self.loc['x'] else 0
            dy = 1 if avg_y > self.loc['y'] else -1 if avg_y < self.loc['y'] else 0

            elevation_diff = self.calculate_elevation_effect(dx, dy)

            self.att['res'] -= abs(elevation_diff)

            # Update location towards the group's average position
            self.loc['x'] += dx
            self.loc['y'] += dy

        # print(f"Moving with group: New location ({self.loc[ 'x' ]}, {self.loc[ 'y' ]})")

    def replenish_resources(self):
        # Check if the current location is a resource point
        if (self.loc['x'], self.loc['y']) in self.grid.resource_points:
            # Increase resources by a random amount up to max_res_gain
            self.att['res'] += random.randint(max_res_gain * .5, max_res_gain)

            # Remove the resource point from the grid
            self.grid.resource_points.remove((self.loc['x'], self.loc['y']))

    def distribute_resources(self):
        # Find adjacent group members
        adjacent_group_members = [entities[member_id] for group in self.grp.values() for member_id in group.members if
                                  member_id in entities and self.is_adjacent(entities[member_id])]

        # Calculate total resources of self and adjacent group members
        total_resources = self.att['res'] + sum(member.att['res'] for member in adjacent_group_members)

        # Distribute resources evenly among self and adjacent group members
        if adjacent_group_members:
            distributed_resource = total_resources / (len(adjacent_group_members) + 1)
            self.att['res'] = round(distributed_resource)
            for member in adjacent_group_members:
                member.att['res'] = distributed_resource

    def turn_into_zombie(self, on_turn_into_zombie_callback):
        new_zombie = Zombie(ttd=start_ttd)
        new_zombie.loc = self.loc.copy()
        new_zombie.id = self.id.replace('_H', '_Z')  # Change the ID suffix from 'H' to 'Z'

        gl.log(self, new_zombie, 'turn', 'zombie')

        if on_turn_into_zombie_callback:
            on_turn_into_zombie_callback(self, new_zombie)

        # remove human from human groups
        for group_id in self.grp.keys():
            if group_id in entities:
                group = entities[group_id]
                group.remove_member(self)

        self.is_z = True
        self.is_h = False

        if self.id in entities:  # Check if the entity is in the dictionary before removing
            entities.pop(self.id)
        entities[new_zombie.id] = new_zombie


class Zombie(Entity):
    def __init__(self, ttd=start_ttd):
        super().__init__(entity_type='Z')
        # super().__init__()
        self.att['ttd'] = ttd
        self.is_h = False
        self.is_z = True
        self.is_active = True
        # global_entities['zombies'].append(self)

    # def __str__(self):
    #     return f"Zombie {self.id}"

    def move(self):
        if self.att['ttd'] <= 0:
            self.is_active = False
            return  # Zombie has run out of time to decay

        old_loc = self.loc.copy()

        if self.grp and self.prob_move_grp():
            self.move_grp()
        elif self.prob_move_prey():
            self.move_prey()
        else:
            self.move_rand()

        ml.log(self, old_loc['x'], old_loc['y'], old_loc['z'], self.loc['x'], self.loc['y'], self.loc['z'])

        self.att['ttd'] -= .5
        # rl.log(entity=self, res_change=-.5, reason='move')

    def prob_move_grp(self):
        collective_res = sum(member.res for member in self.grp.values()) / len(self.grp)
        return collective_res < 5

    def prob_move_prey(self):
        return random.random() < (1 - self.att['ttd'] / 10)

    def move_rand(self):
        # Random movement logic, modifying x and y coordinates
        dx = random.randint(-4, 4)  # Step size for x direction
        dy = random.randint(-4, 4)  # Step size for y direction

        # Check if the new location is within the grid bounds
        new_x = self.loc['x'] + dx
        new_y = self.loc['y'] + dy
        if 0 <= new_x < self.grid.width and 0 <= new_y < self.grid.height:
            elevation_diff = self.calculate_elevation_effect(dx, dy)

            # Subtract resources proportional to the elevation difference
            self.att['res'] -= abs(elevation_diff)

            # Update location
            self.loc['x'] = new_x
            self.loc['y'] = new_y

    def move_grp(self):
        total_members = 0
        total_x = 0
        total_y = 0

        for group in self.grp.values():
            for member_id in group.members:
                member = group.get_member_by_id(member_id)  # Get the Entity object
                if member is not None:
                    total_x += member.loc['x']
                    total_y += member.loc['y']
                    total_members += 1

        if total_members > 0:
            avg_x = total_x / total_members
            avg_y = total_y / total_members

            # Determine the direction to move towards the group's average position
            dx = 1 if avg_x > self.loc['x'] else -1 if avg_x < self.loc['x'] else 0
            dy = 1 if avg_y > self.loc['y'] else -1 if avg_y < self.loc['y'] else 0

            elevation_diff = self.calculate_elevation_effect(dx, dy)

            self.att['res'] -= abs(elevation_diff)

            # Update location towards the group's average position
            self.loc['x'] += dx
            self.loc['y'] += dy

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

            # print(f"Moving towards prey: New location ({self.loc['x']}, {self.loc['y']})")
        else:
            # If there's no prey within the 5x5 grid, the zombie can move randomly or stay in place
            self.move_rand()


class Group:
    groups = []

    def __init__(self, type):
        self.type = type
        self.id = (f"HG_{id_generator.gen_id('H')}" if type == "human" else f"ZG_{id_generator.gen_id('Z')}")
        entities[self.id] = self
        self.members = []
        Group.groups.append(self)

    def add_member(self, entity):
        self.members.append(entity)

    def remove_member(self, entity):
        for member_id in self.members:
            if member_id in entities:
                member = entities[member_id]
                if member.is_z == entity.is_z and member.id == entity.id:
                    gl.log(self, entity, 'remove', 'decayed')
                    if member in self.members:  # Check if the member is in the list
                        self.members.remove(member)
                    break
                elif not member.is_z == entity.is_z and member.id == entity.id:
                    gl.log(self, entity, 'remove', 'starved')
                    if member in self.members:  # Check if the member is in the list
                        self.members.remove(member)
                    break

    def interact(self, other):
        from events import interact  # Import the interact function from events.py

        if isinstance(other, Group):  # If the other is a group
            for member_id in self.members:
                member = self.get_member_by_id(member_id)  # Get the Entity object
                if member is not None:
                    for other_member_id in other.members:
                        other_member = other.get_member_by_id(other_member_id)
                        if other_member is not None:
                            interact(member, other_member)  # Call the interact function with each pair of entities
        else:  # If the other is an individual entity
            for member_id in self.members:
                member = self.get_member_by_id(member_id)
                if member is not None:
                    interact(ent1=member,
                             ent2=other)  # Call the interact function with each entity and the individual entity

    def get_member_by_id(self, member_id):
        return entities[member_id] if member_id in entities else None

    @classmethod
    def reset_groups(cls):
        cls.groups.clear()
        for group in cls.groups:
            del group  # Delete the group object
