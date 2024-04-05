- modeling-v2/
    - config.py
```
import random

log_path = r"C:\Users\TR\Desktop\results"
# log_path = r"C:\Users\tingram\Desktop\results"

inf_rate = 2

start_res = random.randint(5, 10)

start_ttd = random.randint(5, 10)

ttd_rate = .5

res_lose_rate = 2

max_res_gain = 5

size = 100
grid_size = (size, size)
w, h = size, size
vi = 0.025
vj = 0.025
z = 4

num_humans = 100

num_zombies = 5

epochs = 1

days = 50

loser_survival_rate = 0.25  # The loser keeps % of their resources

loser_death_rate = 0.5  # chance that the loser is killed
```
    - directory_contents.md
    - ents.py
```python
import random

from config import start_res, start_ttd, grid_size, max_res_gain
from env import Grid
# from globals import global_entities
from log import ml, gl, rl, MoveRecord, ResRecord
from util import id_generator

entities = {}  # Dictionary to store all entities
starve_cnt = []
class Entity:
    grid = Grid(grid_size=grid_size)

    def __init__(self, grid=grid, entity_type=''):
        self.id = id_generator.gen_id(entity_type)  # Generate a unique ID for the entity
        entities[self.id] = self  # Add the entity to the global entities dictionary
        print(len(entities))
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
        self.day = 0

    def is_adjacent(self, other):
        dx = abs(self.loc['x'] - other.loc['x'])
        dy = abs(self.loc['y'] - other.loc['y'])
        return dx in [0, 1] and dy in [0, 1]

    def interact(self, other):
        from events import interact  # Import the interact function from events.py
        interact(ent1=self, ent2=other)  # Call the interact function with self and other as arguments

    def update_status(self):
        from events import update_status
        update_status(self)
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
        # super().__init__(entity_type='H')
        super().__init__()
        #replace the self id for the entity related to this human in the entities dict
        self.att['res'] = res
        self.is_h = True
        self.is_z = False
        self.is_active = True
        #append the human to the entities dict
        entities[self.id] = self
        # global_entities['humans'].append(self)

    def __str__(self):
        return f"Human {self.id}"

    def move(self):
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
            self.day += 1
            rl.log(entity=self, res_change=-.5, reason='move')

            # Check if the human has run out of resources

            if self.att['res'] <= 0:
                # print(f"Human {self.id} has run out of resources.")
                starve_cnt.append(self.id)
                self.turn_into_zombie()
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

    def turn_into_zombie(self):

        new_zombie = Zombie(ttd=start_ttd)
        new_zombie.loc = self.loc.copy()
        new_zombie.id = self.id.replace('_H', '_Z')  # Change the ID suffix from 'H' to 'Z'
        entities[new_zombie.id] = new_zombie

        # if self.id in entities:
        #     entities.pop(self.id)
        # else:
            # print(f"Entity with ID {self.id} not found in entities dictionary.")


        #remove human from human groups
        for group_id in self.grp.keys():
            if group_id in entities:
                group = entities[group_id]
                group.remove_member(self)

        gl.log(self, new_zombie, 'turn', 'zombie')
        self.is_z = True
        self.is_h = False
        #remove human from humans list
        # simulation.humans.remove(self)
        # simulation.zombies.append(new_zombie)


class Zombie(Entity):
    def __init__(self, ttd=start_ttd):
        # super().__init__(entity_type='Z')
        super().__init__()
        self.att['ttd'] = ttd
        self.is_h = False
        self.is_z = True
        self.is_active = True
        # global_entities['zombies'].append(self)

    def __str__(self):
        return f"Zombie {self.id}"

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
        self.day += 1

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
                    gl.log(self, entity, 'remove', 'turned' if entity.is_z else 'died')
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
                    interact(member, other)  # Call the interact function with each entity and the individual entity

    def get_member_by_id(self, member_id):
        return entities[member_id] if member_id in entities else None

```
    - env.py
```python
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


```
    - events.py
```python
import random

from config import inf_rate, res_lose_rate, ttd_rate
from ents import Group, entities
from log import EncRecord, ResRecord, GrpRecord, el, rl, gl


def interact(simulation, ent1, ent2):
    if not ent1.is_adjacent(ent2):
        return

    # Check if both entities are zombies
    if ent1.is_z and ent2.is_z:
        return  # Skip

    if ent1.is_z or ent2.is_z:
        if not ent1.is_z:
            human, zombie = ent1, ent2
        else:
            human, zombie = ent2, ent1

            # Check if the zombie is active before proceeding with the interaction
        if not zombie.is_active:
            return

        if human.loc['x'] - 2 <= zombie.loc['x'] <= human.loc['x'] + 2 and human.loc['y'] - 2 <= zombie.loc['y'] <= \
                human.loc['y'] + 2:
            human_to_zombie(human, zombie)
    else:
        if ent1.loc['x'] - 2 <= ent2.loc['x'] <= ent1.loc['x'] + 2 and ent1.loc['y'] - 2 <= ent2.loc['y'] <= ent1.loc[
            'y'] + 2:
            human_to_human(simulation, ent1, ent2)


def update_status(entity):
    if entity.is_z:
        entity.att['ttd'] -= ttd_rate
        if entity.att['ttd'] <= 0:
            entity.is_active = False
    else:
        if entity.att['res'] > 0 and not entity.is_z:
            entity.att['res'] -= res_lose_rate
        else:
            entity.is_z = True
            entity.att['res'] = 0
            entity.turn_into_zombie()


def love_encounter(human, other):
    amount = (human.att['res'] + other.att['res']) / 2
    human.att['res'] = other.att['res'] = amount
    human.enc['luv'] += 1
    human.xp['luv'] += .5
    other.enc['luv'] += 1
    other.xp['luv'] += .5

    # creates a group between the two entities if they are not already group mates
    for group_id in human.grp.keys():
        if group_id in entities:  # Check if the Group object exists
            group = entities[group_id]  # Get the Group object
            if other.id in group.members:
                break
    else:
        group = Group("human")
        group.members.append(human.id)
        group.members.append(other.id)
        human.grp[group.id] = group
        other.grp[group.id] = group

    # add each others id to their network as a friend
    human.net['friend'][other.id] = other
    other.net['friend'][human.id] = human

    #  encounter and resource change logging placeholder
    er = EncRecord(human, other, 'love')
    el.logs.append(er)

    rr = ResRecord(human, amount, 'love')
    rl.logs.append(rr)


def war_encounter(simulation, human, other):
    from config import loser_survival_rate, loser_death_rate

    # Determine the winner and loser based on war_xp
    winner, loser = (human, other) if human.xp['war'] > other.xp['war'] else (other, human)

    # Update war_xp and resources
    winner.xp['war'] += .5
    winner.enc['war'] += 1

    er = EncRecord(human, other, 'war')
    el.logs.append(er)

    winner.att['res'] += loser.att['res'] * (1 - loser_survival_rate)

    rr = ResRecord(human, loser.att['res'] * (1 - loser_survival_rate), 'war')
    rl.logs.append(rr)

    loser.att['res'] *= loser_survival_rate
    loser.enc['war'] += 1

    rr = ResRecord(other, -loser.att['res'] * loser_survival_rate, 'war')
    rl.logs.append(rr)

    # Check if loser is dead and handle accordingly
    if loser.att['res'] <= 0 or random.random() < loser_death_rate:
        loser.att['res'] = 0
        loser.turn_into_zombie()
        simulation.zombies.append(loser)
        simulation.humans.remove(loser)
        for group in loser.grp.values():
            group.remove_member(human)
            gr = GrpRecord(group, loser.id, 'remove', 'war')
            gl.logs.append(gr)
    else:
        # Add each other to their network as foes
        winner.net['foe'][loser.id] = loser
        rr = ResRecord(human, loser.att['res'], 'war')
        rl.logs.append(rr)

        loser.net['foe'][winner.id] = winner
        rr = ResRecord(other, loser.att['res'], 'war')
        rl.logs.append(rr)


def theft_encounter(human, other):
    # Determine the winner and loser based on theft_xp
    winner, loser = (human, other) if human.xp['rob'] > other.xp['rob'] else (other, human)

    # Update theft_xp and resources
    winner.xp['rob'] += .5
    er = EncRecord(winner, loser, 'rob')
    el.logs.append(er)

    winner.att['res'] += loser.att['res'] * (loser.xp['rob'] / winner.xp['rob'])
    rr = ResRecord(winner, loser.att['res'] * (loser.xp['rob'] / winner.xp['rob']), 'rob')
    rl.logs.append(rr)

    loser.att['res'] -= loser.att['res'] * (loser.xp['rob'] / winner.xp['rob'])
    rr = ResRecord(loser, -loser.att['res'] * (loser.xp['rob'] / winner.xp['rob']), 'rob')
    rl.logs.append(rr)

    # Check if loser is dead and handle accordingly
    if loser.att['res'] <= 0:
        loser.is_zombie = True
        for group in loser.grp.values():
            group.remove_member(loser)
            gr = GrpRecord(group, loser.id, 'remove', 'rob')
            gl.logs.append(gr)
    else:
        # Add each other to their network as foes
        winner.net['foe'][loser.id] = loser
        loser.net['foe'][winner.id] = winner

    # encounter and resource change logging placeholder


def kill_zombie_encounter(human, zombie):
    human.xp['war'] += .5

    er = EncRecord(human, zombie, 'kill')
    el.logs.append(er)

    human.att['res'] += 2

    rr = ResRecord(human, 2, 'kill')
    rl.logs.append(rr)

    zombie.is_active = False

    for group in zombie.grp.values():
        group.remove_member(zombie)
        gr = GrpRecord(group, zombie.id, 'remove', 'kill')
        gl.logs.append(gr)

    # log


def infect_human_encounter(human, zombie):
    # Create a new zombie group and add the infected human to it
    zombie_group = Group("zombie")
    zombie_group.add_member(human)
    gr = GrpRecord(zombie_group, human.id, 'add', 'infect')
    gl.logs.append(gr)

    human.att['ttd'] = 10
    er = EncRecord(zombie, human, 'infect')
    el.logs.append(er)
    zombie.att['ttd'] += 2
    zombie.enc['inf'] += 1

    # Remove the human from all his current groups
    for group in human.grp.values():
        group.remove_member(human)
        gr = GrpRecord(group, human.id, 'remove', 'infect')
        gl.logs.append(gr)

    human.turn_into_zombie()


def human_to_human(simulation, human, other):
    outcome = random.choices(population=['love', 'war', 'rob', 'run'],
                             weights=[abs(human.xp['luv'] + other.xp['luv'] + 0.1),
                                      abs(human.xp['war'] + other.xp['war'] + 0.1),
                                      abs(human.xp['rob'] + other.xp['rob'] + 0.1),
                                      abs(human.xp['esc'] + other.xp['esc'] + 0.1)
                                      ])
    if outcome[0] == 'love':
        love_encounter(human, other)
    elif outcome[0] == 'war':
        war_encounter(simulation, human, other)
    elif outcome[0] == 'rob':
        theft_encounter(human, other)
    elif outcome[0] == 'esc':
        human.xp['esc'] += .5
        other.xp['esc'] += .25
        er = EncRecord(human, other, 'esc')
        el.logs.append(er)


def human_to_zombie(human, zombie):
    outcome = random.choices(population=['kill',
                                         'inf',
                                         'esc'],
                             weights=[abs(human.xp['war'] + 0.1),
                                      abs(inf_rate + 0.1),
                                      abs(human.xp['esc'] + 0.1)
                                      ])
    if outcome[0] == 'kill':
        kill_zombie_encounter(human, zombie)
        human.xp['war'] += .5
    elif outcome[0] == 'inf':
        infect_human_encounter(human, zombie)
    elif outcome[0] == 'esc':
        human.xp['esc'] += .5
        er = EncRecord(human, zombie, 'esc')
        el.logs.append(er)


def zombie_to_human(zombie, other):
    inf_event = random.choices([True, False],
                               [inf_rate, (1 - inf_rate) + other.xp['esc'] + 0.1])
    if inf_event[0]:
        infect_human_encounter(other, zombie)
    else:
        other.xp['esc'] += .5
        er = EncRecord(other, zombie, 'esc')
        el.logs.append(er)

```
    - globals.py
```python
import pandas as pd
from ents import entities

# Create a dictionary to store the entities by type
global_entities = {'humans': [], 'zombies': [], 'groups': [], 'removed': []}

# Iterate over all entities
for entity in entities.values():
    # Add the entity to the corresponding list in the global_entities dictionary
    if entity.is_h:
        global_entities['humans'].append(str(entity))
    elif entity.is_z:
        global_entities['zombies'].append(str(entity))
    elif not entity.is_active:
        global_entities['removed'].append(str(entity))
    # Add more conditions here if there are more types of entities

# Convert the global_entities dictionary to a DataFrame
# global_entities_df = pd.DataFrame.from_dict(global_entities, orient='index').transpose()
#
# # Set the display options
# pd.set_option('display.max_columns', None)  # Show all columns
# pd.set_option('display.max_rows', None)  # Show all rows
# pd.set_option('display.width', None)  # No max width
# pd.set_option('display.max_colwidth', None)  # Show full width of showing strings

# Print the DataFrame
# print(global_entities_df)
# def print_removed_entities():
#     removed_entities_str = [str(entity) for entity in global_entities['removed']]
#     return removed_entities_str

    def reset_simulation(self):
        entities.clear()
```
    - log.py
```python
import pandas as pd

from env import DayTracker, Epoch


class EncRecord:
    def __init__(self, ent, other, action):
        self.epoch = Epoch.get_current_epoch()
        self.day = DayTracker.get_current_day()
        self.ent_id = ent.id
        # self.ent_loc = ent.loc
        self.other_id = other.id
        # self.other_loc = other.loc
        self.action = action

    def __dict__(self):
        return {'Epoch': self.epoch, 'Day': self.day, 'Entity 1': self.ent_id, 'Entity 2': self.other_id, 'Interaction Type': self.action}

    def __str__(self):
        return f"{self.epoch},{self.day},{self.ent_id},{self.other_id},{self.action}"


class MoveRecord:
    def __init__(self, ent, old_loc_x, old_loc_y, old_loc_z, new_loc_x, new_loc_y, new_loc_z):
        self.epoch = Epoch.get_current_epoch()
        self.day = DayTracker.get_current_day()
        self.entity_id = ent.id
        self.old_loc_x = old_loc_x
        self.old_loc_y = old_loc_y
        self.old_loc_z = old_loc_z
        self.new_loc_x = new_loc_x
        self.new_loc_y = new_loc_y
        self.new_loc_z = new_loc_z

    def __str__(self):
        return f"{self.epoch},{self.day},{self.entity_id},{self.old_loc_x},{self.old_loc_y},{self.old_loc_z},{self.new_loc_x},{self.new_loc_y},{self.new_loc_z}"


class ResRecord:
    def __init__(self, ent, res_change=0, reason=None):
        self.epoch = Epoch.get_current_epoch()
        self.day = DayTracker.get_current_day()
        self.entity_id = ent.id
        # self.entity_loc = ent.loc
        self.res_change = res_change
        self.current_res = ent.att['res']
        self.reason = reason

    def __str__(self):
        return f"{self.epoch},{self.day},{self.entity_id},{self.res_change},{self.current_res},{self.reason}"


class GrpRecord:
    def __init__(self, grp, member, action, reason):
        self.epoch = Epoch.get_current_epoch()
        self.day = DayTracker.get_current_day()
        self.group_id = grp.id
        self.member_id = member
        self.action = action
        self.reason = reason

    def __str__(self):
        return f"{self.epoch},{self.day},{self.group_id},{self.member_id},{self.action},{self.reason}"


class EncLog:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EncLog, cls).__new__(cls)
            cls._instance.logs = []
        return cls._instance

    def log(self, ent, other, action):
        record = EncRecord(ent, other, action)
        self.logs.append(record)

    def display_logs(self):
        for log in self.logs:
            print(str(log))


class MoveLog:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MoveLog, cls).__new__(cls)
            cls._instance.logs = []
        return cls._instance

    def log(self, entity, old_loc_x, old_loc_y, old_loc_z, new_loc_x, new_loc_y, new_loc_z):
        record = MoveRecord(entity, old_loc_x, old_loc_y, old_loc_z, new_loc_x, new_loc_y, new_loc_z)
        self.logs.append(record)

    def display_logs(self):
        for log in self.logs:
            print(str(log))


class ResLog:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResLog, cls).__new__(cls)
            cls._instance.logs = []
        return cls._instance

    def log(self, entity, res_change, reason):
        record = ResRecord(entity, res_change, reason)
        self.logs.append(record)

    def display_logs(self):
        for log in self.logs:
            print(str(log))


class GrpLog:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GrpLog, cls).__new__(cls)
            cls._instance.logs = []
        return cls._instance

    def log(self, group, member, action, reason):
        record = GrpRecord(group, member, action, reason)
        self.logs.append(record)

    def display_logs(self):
        for log in self.logs:
            print(str(log))


# class MetLog:
#     _instance = None
#
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super(MetLog, cls).__new__(cls)
#             cls._instance.__init__()
#         return cls._instance
#
#     def __init__(self):
#         self.metrics = pd.DataFrame(columns=['Epoch', 'Total_Days', 'Ending_Num_Humans', 'Ending_Num_Zombies', 'Peak_Zombies', 'Peak_Groups', 'Love', 'War', 'Rob', 'Esc', 'Kill', 'Infect'])
#         self.logs = []
#
#     def log_metrics(self, epoch, total_days, ending_num_humans, ending_num_zombies, peak_zombies, peak_groups, love, war, rob, esc, kill, infect):
#         self.metrics = self.metrics.append({'Epoch': epoch, 'Total_Days': total_days, 'Ending_Num_Humans': ending_num_humans, 'Ending_Num_Zombies': ending_num_zombies, 'Peak_Zombies': peak_zombies, 'Peak_Groups': peak_groups, 'Love': love, 'War': war, 'Rob': rob, 'Esc': esc, 'Kill': kill, 'Infect': infect}, ignore_index=True)
#
#     def write_to_csv(self, log_path, filename):
#         self.metrics.to_csv(log_path + filename, index=False)


ml = MoveLog()
rl = ResLog()
el = EncLog()
gl = GrpLog()
# metl = MetLog()


```
    - main.py
```python
import os
import pprint

import pandas as pd
from datetime import datetime

from globals import global_entities
from sim import Simulation
from log import ml, el, rl, gl
from config import log_path
# from globals import reset_simulation



def main():
    # Initialize the simulation
    # reset_simulation()
    sim = Simulation()

    # Run the simulation and get the metrics dictionary
    metrics_dict = sim.run()

    # Convert the metrics dictionary to a DataFrame
    metrics_df = pd.DataFrame(metrics_dict, index=[0])

    # Generate a unique folder name using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_log_path = os.path.join(log_path, timestamp)

    # Create the new folder
    os.makedirs(new_log_path, exist_ok=True)

    # Create dataframes for each log and write them to CSV files in the new folder
    move_df = pd.DataFrame([str(record).split(',') for record in ml.logs],
                           columns=["Epoch", "Day", "Entity", "Old X", "Old Y", "Old Z", "New X", "New Y", "New Z"])
    move_df.to_csv(os.path.join(new_log_path, "move_log.csv"), index=False)

    enc_df = pd.DataFrame([str(record).split(',') for record in el.logs],
                          columns=["Epoch", "Day", "Entity 1", "Entity 2", "Interaction Type"])
    enc_df.to_csv(os.path.join(new_log_path, "enc_log.csv"), index=False)

    res_df = pd.DataFrame([str(record).split(',') for record in rl.logs],
                          columns=["Epoch", "Day", "Entity", "Resource Change", "Current Resources", "Reason"])
    res_df.to_csv(os.path.join(new_log_path, "res_log.csv"), index=False)

    grp_df = pd.DataFrame([str(record).split(',') for record in gl.logs],
                          columns=["Epoch", "Day", "Group", "Entity", "Action", "Reason"])
    grp_df.to_csv(os.path.join(new_log_path, "grp_log.csv"), index=False)

    # Write the metrics DataFrame to a CSV file
    metrics_df.to_csv(os.path.join(new_log_path, "metrics_log.csv"), index=False)


    # for entity_type, entities in global_entities.items():
    #     global_entities[entity_type] = [str(entity) for entity in entities if
    #                                     entity.day == max(entity.day for entity in entities)]
    #
    # global_entities_df = pd.DataFrame.from_dict(global_entities, orient='index').transpose()
    # pd.set_option('display.max_columns', None)  # Show all columns
    # pd.set_option('display.max_rows', None)  # Show all rows
    # pd.set_option('display.width', None)  # No max width
    # pd.set_option('display.max_colwidth', None)  # Show full width of showing strings
    # print(global_entities_df)
    # the number of zombies from global_entities dict
    # print(len(global_entities['zombies']))
    return metrics_df, move_df, enc_df, res_df, grp_df


if __name__ == '__main__':
    metrics_df, move_df, enc_df, res_df, grp_df = main()
```
    - markdown_gen.py
```python
import os

# Function to write directory contents to a Markdown file
def write_directory_to_md(dir_path, output_file):
    with open(output_file, 'w') as md_file:
        for root, dirs, files in os.walk(dir_path):
            level = root.replace(dir_path, '').count(os.sep)
            indent = ' ' * 4 * level
            md_file.write(f'{indent}- {os.path.basename(root)}/\n')
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                md_file.write(f'{subindent}- {f}\n')
                # Check if the file is a Python script
                if f.endswith('.py'):
                    # Open the Python script and read its contents
                    with open(os.path.join(root, f), 'r') as script_file:
                        script_contents = script_file.read()
                    # Write the script contents to the markdown file
                    md_file.write(f'```python\n{script_contents}\n```\n')

# Specify the directory path and output Markdown file name
directory_path = directory_path = r'C:\Users\TR\Desktop\Spring 2024 Courses\z\GIT\modeling-v2\modeling-v2'  # Current directory. Change it if necessary  # Current directory. Change it if necessary
output_md = 'directory_contents.md'

# Call the function
write_directory_to_md(directory_path, output_md)

```
    - README.md
    - sim.py
```python
import pandas as pd

from env import Grid, DayTracker, Epoch
from ents import Human, Zombie, Group, entities
from config import grid_size, num_humans, num_zombies, epochs, days
from events import interact
from log import el


class Simulation:

    def __init__(self, humans=num_humans, zombies=num_zombies, e=epochs, d=days):
        self.grid = Grid(grid_size=grid_size)
        self.humans = [Human() for _ in range(humans)]
        self.zombies = [Zombie() for _ in range(zombies)]
        self.epochs = e
        self.days = d

        for human in self.humans:
            self.grid.add_ent(human)
        for zombie in self.zombies:
            self.grid.add_ent(zombie)

    def simulate_day(self):
        for human in list(self.humans):
            human.move()
            human.update_status()
        for zombie in list(self.zombies):
            zombie.move()
            zombie.update_status()

        self.grid.remove_inactive_ents()

        DayTracker.increment_day()

        # Interaction between humans and zombies
        for human in list(self.humans):  # Create a copy of the list
            for zombie in self.zombies:
                interact(self, human, zombie)

        if not self.humans:
            print("Simulation stopped: All entities are zombies.")
            return

        # Interaction between humans
        if self.humans:  # Check if the list is not empty
            humans_copy = list(self.humans)  # Create a copy of the list
            for i in range(len(humans_copy)):
                # print(f"Before interaction: Length of humans_copy = {len(humans_copy)}, i = {i}")
                for j in range(i + 1, len(humans_copy)):
                    interact(self, humans_copy[i], humans_copy[j])
                # print(f"After interaction: Length of humans_copy = {len(humans_copy)}, i = {i}")

        # Interaction between zombies
        # if self.zombies:  # Check if the list is not empty
        #     for i in range(len(self.zombies)):
        #         for j in range(i + 1, len(self.zombies)):
        #             interact(self, self.zombies[i], self.zombies[j])

    def reset_entitiies(self):
        entities.clear()

    def run(self):
        # Initialize the dictionary for metrics
        metrics = {}

        for epoch in range(self.epochs):
            peak_zombies = 0
            peak_groups = 0
            for _ in range(self.days):
                self.simulate_day()
                if len(self.humans) == 0:
                    print("Simulation stopped: All entities are zombies.")
                    return
                self.reset_entitiies()
                peak_zombies = max(peak_zombies, len(self.zombies))
                peak_groups = max(peak_groups, len(Group.groups))

            # Epoch.increment_epoch()

            # Get the final counts of humans and zombies
            ending_num_humans = len(self.humans)
            ending_num_zombies = len(self.zombies)

            # Calculate encounter types from entity attributes
            enc_types = {'love': 0, 'war': 0, 'rob': 0, 'esc': 0, 'kill': 0, 'infect': 0}
            for log in el.logs:
                enc_types[log.action] += 1

            # Log the metrics for this epoch
            metrics['Epoch'] = epoch
            metrics['Total_Days'] = self.days
            metrics['Ending_Num_Humans'] = ending_num_humans
            metrics['Ending_Num_Zombies'] = ending_num_zombies
            metrics['Peak_Zombies'] = peak_zombies
            metrics['Peak_Groups'] = peak_groups
            metrics.update(enc_types)

        # Return the metrics dictionary
        return metrics

```
    - surface_noise.py
```python
#########################################################
"""

Title: surface_noise.py
Author: TR Ingram
Description:


"""
#########################################################

import numpy as np
import perlin as p
from config import vi, vj, z, w, h

#generate a 2d grid of perlin noise that is 20 by 20
def generate_noise(w, h, vi, vj, z):
    noise = p.Perlin(14)
    grid = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            grid[i, j] = noise.noise(i*vi, j*vj, z)
    return grid

# noise = p.Perlin(14)
# w=100
# h=100
# grid = np.zeros((w, h))
# for i in range(w):
#     for j in range(h):
#         grid[i,j] = noise.noise(i*0.1, j*0.2, 0)
#
# grid = generate_noise(100, 100, 0.025, 0.025, 4)
#
# # plot grid
# plt.imshow(grid, cmap='terrain')
# plt.colorbar()
# plt.title('2D Perlin Noise')
# plt.show()


```
    - util.py
```python
import datetime
import random
from config import log_path
import csv
import os

from log import MoveRecord, EncRecord, ResRecord, GrpRecord


class IDGenerator:
    def __init__(self):
        self.characters = "123456789ABCDEFG"
        self.used_ids = set()

    def gen_id(self, entity_type):
        while True:
            new_id = ''.join(random.choices(self.characters, k=6))
            if new_id not in self.used_ids:
                self.used_ids.add(new_id)
                return new_id + "_" +entity_type


# Create a global instance of IDGenerator
id_generator = IDGenerator()


def write_logs_to_csv(log, log_type):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # Get the current date and time
    dir_path = os.path.join(log_path, timestamp)  # Append the timestamp to the log_path
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, log_type + "_log.csv")

    # Open the file in append mode. This will create the file if it doesn't exist.
    with open(file_path, 'a+', newline='') as file:
        writer = csv.writer(file)
        if log_type == "move":
            writer.writerow(
                ["Epoch", "Day", "Entity", "Old Location X", "Old Location Y", "Old Z", "New Location X", "New Location Y", "New Z"])
            for record in log.logs:
                writer.writerow(
                    str(record).split(','))  # Convert the record object to a string and split it into a list
        elif log_type == "enc":
            writer.writerow(["Epoch", "Day", "Entity 1", "Entity 2", "Interaction Type"])
            for record in log.logs:
                writer.writerow(
                    str(record).split(','))  # Convert the record object to a string and split it into a list
        elif log_type == "res":
            writer.writerow(["Epoch", "Day", "Entity", "Resource Change", "Reason"])
            for record in log.logs:
                writer.writerow(
                    str(record).split(','))  # Convert the record object to a string and split it into a list
        elif log_type == "grp":
            writer.writerow(["Epoch", "Day", "Group", "Entity", "Action", "Reason"])
            for record in log.logs:
                writer.writerow(str(record).split(','))  # Convert the record object to a string and split it into a list
        elif log_type == "metrics":
            log.write_to_csv(log_path, log_type + "_log.csv")

```
    - .git/
        - COMMIT_EDITMSG
        - config
        - description
        - FETCH_HEAD
        - HEAD
        - index
        - ORIG_HEAD
        - packed-refs
        - hooks/
            - applypatch-msg.sample
            - commit-msg.sample
            - fsmonitor-watchman.sample
            - post-update.sample
            - pre-applypatch.sample
            - pre-commit.sample
            - pre-merge-commit.sample
            - pre-push.sample
            - pre-rebase.sample
            - pre-receive.sample
            - prepare-commit-msg.sample
            - push-to-checkout.sample
            - update.sample
        - info/
            - exclude
        - logs/
            - HEAD
            - refs/
                - heads/
                    - main
                    - split
                    - v1
                - remotes/
                    - origin/
                        - HEAD
                        - main
                        - split
        - objects/
            - 00/
                - 2574a915325b51d370f879b98d54a7de7e3458
                - c991949e261425425e0beeee48ec1795a317ba
                - ef93bdd7040d5bcd868ab3d90aed3c75692371
            - 02/
                - 5a88df7250ab22f5e54140b7f9dc5fe73ca0b3
            - 03/
                - a92f50636b971ad3598eb6ad75814c610fbacd
            - 04/
                - 4aa3ae3de0ba01258d81fea2b2e9fbf835b076
            - 08/
                - a264d3f0a8394e23f6b770b99df935333a89be
                - bb364193e9a8149353a28a7319e6d48fb0849d
            - 10/
                - 5ce2da2d6447d11dfe32bfb846c3d5b199fc99
            - 13/
                - d7a80695897116938644e12ba1510173e8c88b
            - 14/
                - d27a967a30ce984cfc88dec94c3dfe163b9981
                - f70dae1fe13ea7156415d7174c6a8fafd02fde
            - 16/
                - 9519aabe7011277f1212f4244022ed6fd10616
            - 18/
                - 3fe96009531679796c891bf1a8e39a78d92405
            - 19/
                - c377c13a66ec6cd6e60095d05c05ca3c09c75f
            - 1a/
                - 344d11b825804022af69c0520c000c3036ee00
            - 1b/
                - 46d16692bf59583ee5ecdc12d2017d27e99491
                - 579ec6c58efa33aaffeb0c2b193d58ba03d473
            - 1d/
                - 43f2a048b2a78e2cefc924fef973ddd4738602
            - 23/
                - f7064cc746bd3bd1fe2fb1e061b3c75addd14c
            - 25/
                - 25a14fb3bb402736394cd75f89cbe6c9b193ec
                - bc791969e85c33e7164f883c2425d26a5c8541
            - 26/
                - d33521af10bcc7fd8cea344038eaaeb78d0ef5
                - f0bb3f2c932933444e2c67cae02208ab9acb10
            - 27/
                - 25801b657ec8587ca33b7ba19e0b8a5260cd1b
            - 2a/
                - 458fc41bcb276b3eb505dacc0f0e944d0bcf4b
                - 8ed459b2d5e2ee60a7cea38962ee46a10f9418
            - 2b/
                - 188c27d4cdc204f04f642a93308ad83bc235ee
                - 6219d2c12e6676543b19a0124c7b8835a8fa49
                - ef947c27b6a0e20eb336455a9a0f81c2fcfbf7
            - 2d/
                - 408e968557e75de742dccaee07722b87ef6f4d
            - 34/
                - cbe62d5f886655375f474fdcfba9b04d7dcea4
            - 35/
                - 614bb70579f1bb51d9c3aeea2849a7a9531744
                - eb1ddfbbc029bcab630581847471d7f238ec53
            - 3b/
                - 20a48026df5c4554b2d589845e0a6664453cca
            - 3d/
                - 7e9834dd78c0a3d93a01e92afe37d71fbe2f70
            - 3e/
                - 172a8223213372d67bf56015ae26482a4ba700
                - 9a72ac11430d298a7017e344002adacfd80505
            - 42/
                - fd3d606fa7dbcbd8c65ca92d1ca8ba82bb4c0a
            - 44/
                - 19cb0866cf3e6a9cf9e57d27cd327d943fa7ff
            - 46/
                - 95b4e96055c08d18de2b43e35193ecce76fe7e
            - 48/
                - a717f1a1ff678fda605c897f38eaa93c66acb7
            - 4a/
                - 35dd60ca769c4bf17ec10ca3713398a601507b
            - 4e/
                - 044f5501ab3017a54c27bbec29e9640deb02a4
            - 4f/
                - e37e013aaad9721243acefaef7dd8e0b729f0a
            - 50/
                - 144baa94e22bbd86b761fb4bd3029cf3c1e5e1
                - b3914a216ecb00040bdc0c107bbd2fabb49e4d
                - b43fa6b70daecec724ff22c626425a71a700c5
                - b8f31e3340fa3b35878c1d6b53ee9638ce81a3
            - 52/
                - f2b6023ad312b80a131204cd4f0d70e1155de2
            - 54/
                - 24baf24af8f610f220372e738cfdbe4db4e086
            - 57/
                - 8a38ed2fa719eb0a92a75af6d67c44b395758e
            - 59/
                - ef01ea57c3941beef7d0fa79a7b534c3137098
            - 5a/
                - 12fa16e84567d0bcb685e23fc2128a46b34cc0
                - 600578536880e10ca7d84fabe3fe54ec7b79c8
            - 5e/
                - 6021aa6f0abc666035845bd5f29dbd385de16c
            - 62/
                - 39c6b5b4bacd4b1731b09ce83970d64e1756ce
                - 83575e41d553ffe59043dbc8873ed9486de22d
            - 68/
                - e9758176ed36d1328469778f85980f1cb7ebd1
            - 6a/
                - e6428878714e50e50dbe4a9ae9ddf261734f02
            - 6d/
                - bd9626629a870c04b43a781f0d144a041fb75e
            - 70/
                - 9c1517c2605c6a72cc0c0100f113594e17b554
            - 73/
                - a4ee362ebc9ee762a0c92482bc62e8b79e2a2c
            - 79/
                - 3058fff90aa3e6d07ff22b7f9656c02232e3e9
                - 5f8e738b25b72be643f8a01749cd78480fafaf
                - 88ab9743d402b6954a921829cda64e4f803c02
                - f4197a1ea2eebf34d975b89c5a2fb50640d0ec
            - 7a/
                - e07313d61262bf8a2c88a68cdf6f2ee1f5da13
            - 7c/
                - ba8b1eba9e9b6d61a1cf8cf667debffba2c22a
            - 7d/
                - 9c73b22797067e4021563c0b1d3e2c3a351f59
            - 81/
                - 4a75c92f3443f292b6712b2ff43004776c6bc6
            - 83/
                - 04d65c3875e1315dc1edced38215a2a4559f5d
            - 84/
                - fc89d4a540de4b2786918ea3cb08cac1549c99
            - 85/
                - 380d4a8a89d21966e8f31110c4e6203e73e5f1
            - 86/
                - c9c9e94299196c876081e7c5e0e1c091dae9e1
                - ce221609a94d87d436173624f1f99d94c931d5
            - 8b/
                - 26b308cad3e54dc0882c4d4087768de1711bdf
                - 8b5f6e5842bfcff446a5918991d66f83127598
                - cf97b891cb6964afc7e846c36c01242012422a
            - 8c/
                - 2db586225325ac9aeee03202ddc63661cb7f79
            - 8e/
                - 72babbc3523e69c0c6f7a8de4fccf73b1d2b85
            - 92/
                - 68048c01f1ceaec38c0c7e9c40cb2bf337e2da
            - 96/
                - 64ff67e8ec5024725e905a87c95f7011ad8415
            - 97/
                - 824ec9b0149e48c91dce4fee530335b327c686
            - a1/
                - cd3f69426b5cbc620bbec5d33c0f6f7dbd8349
            - a4/
                - 0fce18e7c4e01747dbd80f818e9a58ef645d8b
            - a5/
                - d6bc82f40184ee39c6f7625c5b6d6684ff0218
            - a7/
                - 42be609c5d709a0e9ae9c328f9931a7a2d417e
                - b56d315bde8d3554dd4343409ec8d9b4a52f1f
            - aa/
                - 1ee3f538ee16803cb461dd4794a8ca34b7176d
            - ac/
                - 5f770f8d66244d5505dcd0b06ba87c63890d75
            - af/
                - a737ec5014fecbb1fcc4c139d55d5272b60c6e
            - b1/
                - 379203a17cf571a999126e3f16a18e6469fc1a
                - 584c8dfd1bd25fbcdf42d35daa3392f973f622
            - b3/
                - e7d55b499adedb371dfa722fd298bb6681ef76
            - b5/
                - 0ef17abda9d4d02c62d93d8d7ac12b7c4f3279
            - b6/
                - 1f246004108bec41fa942114446ad8f01db2dd
            - b8/
                - 42ef781337ec70a9abb67cb9707c1dfeeee370
            - b9/
                - 82f28772a9c4243f12304f033d927802641a59
            - bb/
                - e3846248c6911f0201588fa599fc69a7a20c55
            - bc/
                - 16fd3d568bd683939f4bd340a63fb98a3f4933
            - bd/
                - 8589b0e38f86fa38b766ed22f6655286c0c014
            - be/
                - 6ec3c3ccfc9d118dc0b19e621dbe62bed8d65b
            - c5/
                - 617538a36ba91612dd72fcf5729295130aec93
                - f712eb94b97446f71c0f87a3533bbd4e58e903
            - c8/
                - 7f62ce320aa716eb81cb5d3ffe6ff38b631c52
            - cb/
                - 6ad897d2b59985deaa553db19ceb1dbdc1961e
            - cc/
                - 41ecc4be2cb2cb43a6ff70432157cb948032bb
            - d0/
                - 876a78d06ac03b5d78c8dcdb95570281c6f1d6
            - d2/
                - 3bd8a9ed638a10a47e8028f5cdb0bdc456397a
                - 539009218ce6463ef119c8b139459e14aee170
                - efed81799af4570a33443798cebb464ffc1bbc
            - d4/
                - 512767bf66ab5fb479fb7da2e741cc4afdd129
                - 9952892357715df5310e5445d16c34078f4300
            - d5/
                - 106ad8c81818aa0d92a6f716e8ff71446deaff
                - 5eca8a0e13ca037885139e9980595ffc65fc34
            - d7/
                - 793b8f78acbac1b4b0916d6901fbd9640203ba
            - d9/
                - 2988e71a71b52ede3b788f4b1222c022170f16
            - da/
                - 7d7c1a1b10f461451db68fa8a5dee845897e0d
                - 9dbcf508b496167a098055fb01c4b9cb58cc2a
                - ede0ddb1eeb617bcdbcd5940a6bb4269c35a41
            - db/
                - 8786c06ed9719181cd25ebe2615cc99c8490eb
            - dc/
                - 83ac4bf1a6f25af9d960ca46c551ee0523adf0
            - e0/
                - 16211007fe1939b3cd037e2270df43613b3d8b
                - 93dcb3dca3a620b76a614618423cfaa15bf14e
            - e6/
                - 6b786e4eddb84fa406a635a776bd6ea35ec39b
                - 9de29bb2d1d6434b8b29ae775ad8c2e48c5391
            - ee/
                - 6a3eb04c98d755c28cb226dc367bd1fb73de35
            - ef/
                - 65f7d75ecfeb64027d22365bfd712063d08fa5
            - f1/
                - 430ba31e236cd38a16f402ede5d06121f5af85
            - f2/
                - d0035d4d2a4e548e5c8743f6fc465151c1b308
            - f9/
                - c64c547d654000c6d12c7360633956a11033d7
            - fa/
                - 254fda7fbd0d4dd622f5f7353787a25c42337d
                - 98e1aed9ef4a9d5e7414f1bb31eacf92f29f13
            - fc/
                - 0221e9c83a98ea27331b34c49b324162e692da
            - fe/
                - 273e7b87d15b6b4fff74dea27c230e23292434
            - info/
            - pack/
                - pack-365c62a4ecbbd282db4574439572ed9cff2dac9c.idx
                - pack-365c62a4ecbbd282db4574439572ed9cff2dac9c.pack
        - refs/
            - heads/
                - main
                - split
                - v1
            - remotes/
                - origin/
                    - HEAD
                    - main
                    - split
            - tags/
    - .idea/
        - .gitignore
        - misc.xml
        - modeling-v2.iml
        - modules.xml
        - vcs.xml
        - inspectionProfiles/
            - profiles_settings.xml
    - __pycache__/
        - config.cpython-312.pyc
        - config.cpython-39.pyc
        - ents.cpython-312.pyc
        - ents.cpython-39.pyc
        - env.cpython-312.pyc
        - env.cpython-39.pyc
        - events.cpython-312.pyc
        - events.cpython-39.pyc
        - globals.cpython-312.pyc
        - log.cpython-312.pyc
        - log.cpython-39.pyc
        - main.cpython-312.pyc
        - sim.cpython-312.pyc
        - sim.cpython-39.pyc
        - surface_noise.cpython-312.pyc
        - surface_noise.cpython-39.pyc
        - util.cpython-312.pyc
        - util.cpython-39.pyc
