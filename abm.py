import random

from config import grid_size, start_res, start_ttd, max_res_gain, ttd_rate, res_lose_rate, inf_rate, w, h, \
    vi, vj, z, num_humans, num_zombies, epochs, days
from surface_noise import generate_noise
from util import id_generator
from network_manager import NetworkManager


entities = {}  # Dictionary to store all entities


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
        removed_ents = 0
        for ent in self.ents:
            if not ent.is_active:
                if (ent.loc['x'], ent.loc['y']) in self.occupied_positions:
                    self.occupied_positions.remove((ent.loc['x'], ent.loc['y']))
                self.ents.remove(ent)
                removed_ents += 1
        return removed_ents

    # def simulate_day(self):
    #     for ent in self.ents:
    #         ent.move(self)
    #         ent.update_status()
    #
    #     self.remove_inactive_ents()
    #
    #     # Increment the day in the simulation
    #     # DayTracker.increment_day()

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


class Simulation:
    current_epoch = 0

    def __init__(self, humans=num_humans, zombies=num_zombies, e=epochs, d=days):
        self.grid = Grid(grid_size=grid_size)
        # self.network_manager = NetworkManager()
        self.humans = [Human() for _ in range(humans)]
        self.zombies = [Zombie() for _ in range(zombies)]
        self.epochs = e
        self.days = d

        # Initialize metrics
        self.starting_num_humans = len(self.humans)
        self.starting_num_zombies = len(self.zombies)
        self.num_starved = 0
        self.num_infected = 0
        self.total_num_zombies = len(self.zombies)
        self.total_removed = 0

        for human in self.humans:
            self.grid.add_ent(human)
            # self.network_manager.add_agent(human.id, 'human')
        for zombie in self.zombies:
            self.grid.add_ent(zombie)
            # self.network_manager.add_agent(zombie.id, 'zombie')

    def simulate_day(self):
        for human in list(self.humans):
            human.move(self)
            human.update_status(self)
            if human.starved:
                self.num_starved += 1
        for zombie in list(self.zombies):
            zombie.move()
            zombie.update_status(self)

        removed_ents = self.grid.remove_inactive_ents()
        self.total_removed += removed_ents

        # Interaction between humans and zombies
        for human in list(self.humans):  # Create a copy of the list
            for zombie in self.zombies:
                interact(self, human, zombie)

        # print(DayTracker.get_current_day(),len(self.humans), len(self.zombies))
        # Interaction between humans
        if self.humans:  # Check if the list is not empty
            humans_copy = list(self.humans)  # Create a copy of the list
            for i in range(len(humans_copy)):
                for j in range(i + 1, len(humans_copy)):
                    interact(self, humans_copy[i], humans_copy[j])

    def handle_turn_into_zombie(self, human, new_zombie):
        # Handle the human to zombie transition within the simulation
        self.humans.remove(human)
        self.zombies.append(new_zombie)
        self.num_infected += 1
        self.total_num_zombies += 1

    def reset_simulation(self):
        # Reset the metrics
        self.starting_num_humans = len(self.humans)
        self.starting_num_zombies = len(self.zombies)
        self.num_starved = 0
        self.num_infected = 0
        self.total_num_zombies = len(self.zombies)
        self.total_removed = 0
        self.enc_types = {'love': 0, 'war': 0, 'rob': 0, 'esc': 0, 'kill': 0, 'infect': 0}

        # Reset the encounter logs
        # el.logs.clear()

        # Reset the DayTracker, Epoch and Group class variables
        DayTracker.reset()
        Group.reset_groups()

    def run(self):
        # Initialize the list for metrics
        metrics_list = []
        encounter_logs = []

        for epoch in range(self.epochs):

            self.reset_simulation()

            Epoch.increment_sim()

            metrics = {}

            peak_zombies = 0
            peak_groups = 0

            for _ in range(self.days):
                DayTracker.increment_day()
                self.simulate_day()

                peak_zombies = max(peak_zombies, len(self.zombies))
                peak_groups = max(peak_groups, len(Group.groups))

                if len(self.humans) == 0 or len(self.zombies) == 0:
                    break

            # Get the final counts of humans and zombies
            ending_num_humans = len(self.humans)
            ending_num_zombies = len(self.zombies)

            # Calculate encounter types from entity attributes
            enc_types = {'love': 0, 'war': 0, 'rob': 0, 'esc': 0, 'kill': 0, 'infect': 0}
            for log in el.logs:
                enc_types[log.action] += 1

            # Log the metrics for this epoch
            metrics['Epoch'] = Epoch.get_current_epoch()
            if len(self.humans) == 0 or len(self.zombies) == 0:
                dd = DayTracker.get_current_day()
                metrics['Day'] = dd
            metrics['Ending_Num_Humans'] = ending_num_humans
            metrics['Ending_Num_Zombies'] = ending_num_zombies
            metrics['Peak_Zombies'] = peak_zombies
            metrics['Peak_Groups'] = peak_groups
            metrics['Starting_Num_Humans'] = self.starting_num_humans
            metrics['Starting_Num_Zombies'] = self.starting_num_zombies
            metrics['Num_Starved'] = self.num_starved
            metrics['Num_Infected'] = self.num_infected
            metrics['Total_Num_Zombies'] = self.total_num_zombies
            metrics['Total_Removed'] = self.total_removed

            metrics.update(enc_types)
            metrics_list.append(metrics)

            self.__init__()

            encounter_logs.append(el.logs.copy())
            el.logs.clear()

        # Return the list of metrics dictionaries
        return metrics_list, encounter_logs


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

    # def interact(self, other):
    #     # from events import interact  # Import the interact function from events.py
    #     interact(ent1=self, ent2=other)  # Call the interact function with self and other as arguments

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
                    member = get_member_by_id(member_id)  # Get the Entity object
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
                member = get_member_by_id(member_id)  # Get the Entity object
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
                member = get_member_by_id(member_id)  # Get the Entity object
                if member is not None:
                    for other_member_id in other.members:
                        other_member = other.get_member_by_id(other_member_id)
                        if other_member is not None:
                            interact(member, other_member)  # Call the interact function with each pair of entities
        else:  # If the other is an individual entity
            for member_id in self.members:
                member = get_member_by_id(member_id)
                if member is not None:
                    interact(ent1=member,
                             ent2=other)  # Call the interact function with each entity and the individual entity

    @classmethod
    def reset_groups(cls):
        cls.groups.clear()
        for group in cls.groups:
            del group  # Delete the group object

# ==================================================================


class DayTracker:
    current_day = 0

    @classmethod
    def increment_day(cls):
        cls.current_day += 1

    @classmethod
    def get_current_day(cls):
        return cls.current_day

    @classmethod
    def reset(cls):
        cls.current_day = 0


class Epoch:
    epoch = 0

    @classmethod
    def increment_sim(cls):
        cls.epoch += 1

    @classmethod
    def get_current_epoch(cls):
        return cls.epoch

    @classmethod
    def reset(cls):
        cls.epoch = 0


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


ml = MoveLog()
rl = ResLog()
el = EncLog()
gl = GrpLog()


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
            human_to_zombie(human, zombie, simulation)
    else:
        if ent1.loc['x'] - 2 <= ent2.loc['x'] <= ent1.loc['x'] + 2 and ent1.loc['y'] - 2 <= ent2.loc['y'] <= ent1.loc[
            'y'] + 2:
            human_to_human(simulation, ent1, ent2)


def update_status(entity, simulation):
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
            entity.turn_into_zombie(on_turn_into_zombie_callback=simulation.handle_turn_into_zombie)


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
        loser.turn_into_zombie(on_turn_into_zombie_callback=simulation.handle_turn_into_zombie)
        if loser in simulation.humans:  # Check if loser is in the list before removing
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


def infect_human_encounter(human, zombie, simulation):
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

    human.turn_into_zombie(on_turn_into_zombie_callback=simulation.handle_turn_into_zombie)


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


def human_to_zombie(human, zombie, simulation):
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
        infect_human_encounter(human, zombie, simulation)
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


def get_member_by_id(member_id):
    return entities[member_id] if member_id in entities else None

# if __name__ == '__main__':
#     metrics_df, move_df, res_df, enc_df, grp_df = main()
