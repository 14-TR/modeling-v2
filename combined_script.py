import random

import numpy as np

from config import grid_size, start_res, start_ttd, ttd_rate, res_lose_rate, inf_rate, w, h, \
    vi, vj, z, num_humans, num_zombies, epochs, days
from network_manager import NetworkManager
from surface_noise import generate_noise
from util import id_generator

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
        self.resource_decay_rate = 0.1  # Rate at which resources decay if not replenished

    def gen_res_pnt(self, num_points):
        points = set()
        while len(points) < num_points:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in points:
                points.add((x, y))
                self.resource_points.add((x, y, 100))  # Adding with initial strength
        return points

    def update_resources(self):
        new_resources = set()
        for x, y, strength in self.resource_points:
            if strength > 0:
                new_strength = max(0, strength - self.resource_decay_rate)
                new_resources.add((x, y, new_strength))
        self.resource_points = new_resources

    def get_xy_res_pnt(self):
        return [(x, y) for x, y, _ in self.resource_points]

    def get_nearest_res_pnt(self, entity_x, entity_y):
        min_distance = float('inf')
        closest_resource_point = None
        for x, y, strength in self.resource_points:
            if strength > 0:
                distance = self.calc_dist(entity_x, entity_y, x, y)
                if distance < min_distance:
                    min_distance = distance
                    closest_resource_point = (x, y)
        return closest_resource_point if closest_resource_point else (0, 0)

    def get_nearest_prey(self, entity):
        min_distance = float('inf')
        closest_prey = None
        min_x, max_x = entity.loc['x'] - 2, entity.loc['x'] + 2
        min_y, max_y = entity.loc['y'] - 2, entity.loc['y'] + 2
        for prey in self.ents:
            if not prey.is_z and min_x <= prey.loc['x'] <= max_x and min_y <= prey.loc['y'] <= max_y:
                distance = self.calc_dist(entity.loc['x'], entity.loc['y'], prey.loc['x'], prey.loc['y'])
                if distance < min_distance:
                    min_distance = distance
                    closest_prey = prey
        return (closest_prey.loc['x'], closest_prey.loc['y']) if closest_prey else None

    @staticmethod
    def calc_dist(x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def add_ent(self, ent):
        while (ent.loc['x'], ent.loc['y']) in self.occupied_positions:
            ent.loc['x'] = random.randint(1, self.width - 1)
            ent.loc['y'] = random.randint(1, self.height - 1)
        self.occupied_positions.add((ent.loc['x'], ent.loc['y']))
        self.ents.append(ent)

    def remove_inactive_ents(self):
        removed_ents = 0
        for ent in self.ents:
            if not ent.is_active:
                self.occupied_positions.remove((ent.loc['x'], ent.loc['y']))
                self.ents.remove(ent)
                removed_ents += 1
        return removed_ents

    def get_elevation(self, x, y):
        return self.surface[x][y]

    def append_surface(self, surface):
        self.surface = surface

    def is_in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def are_adjacent(self, entity, other):
        return abs(entity.loc['x'] - other.loc['x']) <= 1 and abs(entity.loc['y'] - other.loc['y']) <= 1


class Simulation:
    current_epoch = 0

    def __init__(self, humans=num_humans, zombies=num_zombies, e=epochs, d=days):
        self.grid = Grid(grid_size=grid_size)
        self.network_manager = NetworkManager()
        self.humans = [Human() for _ in range(humans)]
        self.zombies = [Zombie() for _ in range(zombies)]
        self.epochs = e
        self.days = d
        self.metrics_list = []

    def simulate_day(self):
        # Process each entity's move and interactions
        for entity in self.humans + self.zombies:
            entity.move()
            self.process_interactions(entity)
            entity.update_status(self)

        # Remove inactive entities and update resource levels
        removed_ents = self.grid.remove_inactive_ents()
        self.grid.update_resources()

        # Track daily metrics and interactions
        self.track_metrics()

    def process_interactions(self, entity):
        # Check for nearby entities and process potential interactions
        for other in self.humans + self.zombies:
            if entity != other and self.grid.are_adjacent(entity, other):
                interact(self, entity, other)

    def track_metrics(self):
        # Enhanced metrics tracking
        metrics = {
            'num_humans': len([h for h in self.humans if h.is_active]),
            'num_zombies': len([z for z in self.zombies if z.is_active]),
            'resources': sum(res for _, res in self.grid.resource_points),
            'day': DayTracker.current_day
        }
        self.metrics_list.append(metrics)

    def run(self):
        # Simulate over defined epochs and days
        for epoch in range(self.epochs):
            self.reset_simulation()
            Epoch.increment_sim()
            while not self.is_epoch_over():
                DayTracker.increment_day()
                self.simulate_day()

    def is_epoch_over(self):
        # End epoch if no humans or zombies are left active
        return not self.humans or not self.zombies

    def reset_simulation(self):
        # Reset simulation parameters for a new epoch
        self.grid = Grid(grid_size=grid_size)
        self.network_manager = NetworkManager()
        self.humans = [Human() for _ in range(num_humans)]
        self.zombies = [Zombie() for _ in range(num_zombies)]
        DayTracker.reset()


class Entity:
    grid = Grid(grid_size=grid_size)

    def __init__(self, grid=grid, entity_type='', network_manager=None):
        self.id = id_generator.gen_id(entity_type)  # Generate a unique ID for the entity
        entities[self.id] = self  # Add the entity to the global entities dictionary
        # print(len(entities))
        self.network_manager = network_manager
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
        self.att['res'] = res
        self.is_h = True
        self.is_z = False
        self.is_active = True
        entities[self.id] = self
        self.starved = False

        # Q-Learning attributes
        self.q_table = np.zeros((self.grid.width * self.grid.height, 4))  # Actions: 0=up, 1=down, 2=left, 3=right
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01

    def __str__(self):
        return f"Human {self.id}"

    def get_state(self):
        return self.grid.width * self.loc['y'] + self.loc['x']

    def choose_action(self):
        state = self.get_state()
        if random.random() < self.exploration_rate:
            return random.randint(0, 3)  # Random action
        else:
            return np.argmax(self.q_table[state])  # Best known action from Q-table

    def update_q_table(self, old_state, action, reward, new_state):
        old_value = self.q_table[old_state, action]
        future_rewards = np.max(self.q_table[new_state])
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * future_rewards - old_value)
        self.q_table[old_state, action] = new_value

    def move(self):
        old_state = self.get_state()
        action = self.choose_action()
        reward = self.perform_action(action)
        new_state = self.get_state()
        self.update_q_table(old_state, action, reward, new_state)
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)

        self.replenish_resources()
        self.distribute_resources()
        self.att['res'] -= 0.5  # Simulate resource depletion over time

        if self.att['res'] <= 0:
            self.turn_into_zombie()

    def perform_action(self, action):
        actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Mapping actions to movements: up, down, left, right
        dx, dy = actions[action]
        new_x = self.loc['x'] + dx
        new_y = self.loc['y'] + dy
        if self.grid.is_in_bounds(new_x, new_y):
            self.loc['x'] = new_x
            self.loc['y'] = new_y
            if (new_x, new_y) in self.grid.resource_points:
                return 10  # Reward for finding a resource
        return -1  # Penalty for moving

    def replenish_resources(self):
        if (self.loc['x'], self.loc['y']) in self.grid.resource_points:
            self.att['res'] += random.randint(1, 10)  # Simulate resource collection
            self.grid.resource_points.remove((self.loc['x'], self.loc['y']))

    def distribute_resources(self):
        if self.grp:
            total_res = self.att['res']
            members = [entities[m_id] for m_id in self.grp if m_id in entities and entities[m_id].is_adjacent(self)]
            if members:
                total_res += sum(m.att['res'] for m in members)
                avg_res = total_res / (len(members) + 1)
                self.att['res'] = avg_res
                for m in members:
                    m.att['res'] = avg_res

    def turn_into_zombie(self, inf_by=None):
        new_zombie = Zombie(ttd=start_ttd)
        new_zombie.loc = self.loc.copy()
        new_zombie.id = self.id.replace('_H', '_Z')
        entities.pop(self.id)
        entities[new_zombie.id] = new_zombie
        self.is_z = True
        self.is_h = False


class Zombie(Entity):
    def __init__(self, ttd=start_ttd):
        super().__init__(entity_type='Z')
        self.att['ttd'] = ttd
        self.is_h = False
        self.is_z = True
        self.is_active = True
        entities[self.id] = self

    def __str__(self):
        return f"Zombie {self.id}"

    def move(self):
        if self.att['ttd'] <= 0:
            self.is_active = False
            return  # Zombie decays when time to decay reaches zero

        old_loc = self.loc.copy()
        action = self.choose_action()
        self.perform_action(action)

        ml.log(self, old_loc['x'], old_loc['y'], old_loc['z'], self.loc['x'], self.loc['y'], self.loc['z'])

        self.att['ttd'] -= 0.5  # Time-to-decay decreases after each move

    def choose_action(self):
        # Zombies move randomly unless prey is nearby
        if self.prob_move_prey():
            return 'prey'
        elif random.random() < 0.5:
            return 'random'
        else:
            return 'group'

    def perform_action(self, action):
        if action == 'prey':
            self.move_prey()
        elif action == 'group':
            self.move_grp()
        else:
            self.move_rand()

    def prob_move_prey(self):
        # Probability of moving towards prey decreases as the time-to-decay decreases
        return random.random() < (self.att['ttd'] / start_ttd)

    def move_rand(self):
        dx, dy = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])  # Random movement in four directions
        self.update_position(dx, dy)

    def move_grp(self):
        if not self.grp:
            return
        avg_x, avg_y = self.calculate_group_center()
        self.update_position_toward_target(avg_x, avg_y)

    def move_prey(self):
        prey_coords = self.grid.get_nearest_prey(self)
        if prey_coords:
            self.update_position_toward_target(*prey_coords)

    def update_position(self, dx, dy):
        new_x = self.loc['x'] + dx
        new_y = self.loc['y'] + dy
        if self.grid.is_in_bounds(new_x, new_y):
            self.loc['x'], self.loc['y'] = new_x, new_y

    def update_position_toward_target(self, target_x, target_y):
        dx = np.sign(target_x - self.loc['x'])
        dy = np.sign(target_y - self.loc['y'])
        self.update_position(dx, dy)

    def calculate_group_center(self):
        total_x, total_y, count = 0, 0, 0
        for member_id in self.grp:
            member = entities.get(member_id)
            if member:
                total_x += member.loc['x']
                total_y += member.loc['y']
                count += 1
        return (total_x // count, total_y // count) if count > 0 else (self.loc['x'], self.loc['y'])


class Group:
    groups = []

    def __init__(self, type, network_manager):
        self.type = type
        self.id = (f"HG_{id_generator.gen_id('H')}" if type == "human" else f"ZG_{id_generator.gen_id('Z')}")
        entities[self.id] = self
        self.network_manager = network_manager
        self.members = []
        Group.groups.append(self)

    def add_member(self, entity):
        self.members.append(entity)
        self.network_manager.add_agent(entity.id, 'human' if entity.is_h else 'zombie')

    def remove_member(self, entity):
        for member_id in self.members:
            if member_id in entities:
                member = entities[member_id]
                if member.is_z == entity.is_z and member.id == entity.id:
                    gl.log(self, entity, 'remove', 'decayed')
                    if member in self.members:  # Check if the member is in the list
                        self.members.remove(member)
                    self.network_manager.remove_agent(member.id)
                    break
                elif not member.is_z == entity.is_z and member.id == entity.id:
                    gl.log(self, entity, 'remove', 'starved')
                    if member in self.members:  # Check if the member is in the list
                        self.members.remove(member)
                    self.network_manager.remove_agent(member.id)
                    break

    def interact(self, other):  # Import the interact function from events.py

        if isinstance(other, Group):  # If the other is a group
            for member_id in self.members:
                member = get_member_by_id(member_id)  # Get the Entity object
                if member is not None:
                    for other_member_id in other.members:
                        other_member = get_member_by_id(other_member_id)
                        if other_member is not None:
                            interact(ent1=member,
                                     ent2=other_member)  # Call the interact function with each pair of entities
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
        return {'Epoch': self.epoch, 'Day': self.day, 'Entity 1': self.ent_id, 'Entity 2': self.other_id,
                'Interaction Type': self.action}

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
            human_to_human(ent1, ent2, simulation)


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
            entity.turn_into_zombie()


def love_encounter(human, other, simulation):
    amount = (human.att['res'] + other.att['res']) / 2
    human.att['res'] = other.att['res'] = amount
    human.enc['luv'] += 1
    human.xp['luv'] += .5
    other.enc['luv'] += 1
    other.xp['luv'] += .5

    # creates a group between the two entities if they are not already group mates
    for group_id in human.grp.keys():
        if group_id in entities:  # Check if the Group object exists
            # add edge between the two entities in the network
            human.network_manager.add_edge(human.id, other.id)
            group = entities[group_id]  # Get the Group object
            if other.id in group.members:
                break
    else:
        group = Group("human", network_manager=simulation.network_manager)
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


def war_encounter(human, other, simulation):
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
    zombie_group = Group("zombie", network_manager=simulation.network_manager)
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

    human.turn_into_zombie(inf_by=zombie)


def human_to_human(human, other, simulation):
    outcome = random.choices(population=['love', 'war', 'rob', 'run'],
                             weights=[abs(human.xp['luv'] + other.xp['luv'] + 0.1),
                                      abs(human.xp['war'] + other.xp['war'] + 0.1),
                                      abs(human.xp['rob'] + other.xp['rob'] + 0.1),
                                      abs(human.xp['esc'] + other.xp['esc'] + 0.1)
                                      ])
    if outcome[0] == 'love':
        love_encounter(human, other, simulation)
    elif outcome[0] == 'war':
        war_encounter(human, other, simulation)
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


def zombie_to_human(zombie, other, simulation):
    inf_event = random.choices([True, False],
                               [inf_rate, (1 - inf_rate) + other.xp['esc'] + 0.1])
    if inf_event[0]:
        infect_human_encounter(other, zombie, simulation)
    else:
        other.xp['esc'] += .5
        er = EncRecord(other, zombie, 'esc')
        el.logs.append(er)


def get_member_by_id(member_id):
    return entities[member_id] if member_id in entities else None

# if __name__ == '__main__':
#     metrics_df, move_df, res_df, enc_df, grp_df = main()

import numpy as np
import random
import pandas as pd
import json

from matplotlib import pyplot as plt


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=4)


def load_from_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


class Grid:
    def __init__(self, width, height, num_resources):
        self.width = width
        self.height = height
        self.resources_count = num_resources
        self.resources = self.generate_resources(num_resources)
        self.initial_resources = self.resources.copy()

    def get_resource_positions(self):
        """Returns a list of tuples, each containing the coordinates of a resource."""
        return list(self.resources)

    def generate_resources(self, num_resources):
        return {(random.randint(0, self.width - 1), random.randint(0, self.height - 1)) for _ in range(num_resources)}

    def remove_resource(self, x, y):
        if (x, y) in self.resources:
            self.resources.remove((x, y))
            self.resources_count -= 1

    def get_nearest_res_pnt(self, x, y):
        nearest_resource = None
        min_distance = float('inf')
        for res in self.resources:
            distance = np.sqrt((res[0] - x) ** 2 + (res[1] - y) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_resource = res
        return nearest_resource if nearest_resource else (0, 0)

    def get_distance_to_nearest_res_pnt(self, x, y):
        nearest_resource = self.get_nearest_res_pnt(x, y)
        return np.sqrt((nearest_resource[0] - x) ** 2 + (nearest_resource[1] - y) ** 2)

    def current_resource_count(self):
        return self.resources_count

    def get_initial_resource_positions(self):
        """Returns a list of tuples, each containing the initial coordinates of a resource."""
        return list(self.initial_resources)


class SimpleAgent:
    def __init__(self, id, grid, loc=(0, 0), learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0,
                 exploration_decay=0.995, min_exploration_rate=0.01):
        self.id = id
        self.grid = grid
        self.loc = {'x': loc[0], 'y': loc[1]}
        self.q_table = np.zeros((grid.width * grid.height, 2))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.cumulative_reward = 0
        self.past_rewards = []
        self.history = []
        self.path = []

    def log(self, action, reward):
        self.history.append((action, reward))

    def plot_history(self):
        actions, rewards = zip(*self.history)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(actions)
        plt.xlabel('Episode')
        plt.ylabel('Action')
        plt.title('Agent Actions Over Time')
        plt.subplot(1, 2, 2)
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Agent Rewards Over Time')
        plt.tight_layout()
        plt.show()

    def save_q_table(self, filename):
        np.save(filename, self.q_table)

    def load_q_table(self, filename):
        self.q_table = np.load(filename)

    def get_state(self):
        return self.grid.width * self.loc['y'] + self.loc['x']

    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return random.choice([0, 1])
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        future_rewards = np.max(self.q_table[next_state])
        estimated_q = self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * (
                reward + self.discount_factor * future_rewards - estimated_q)

    def move_towards_resource(self):
        nearest_res = self.grid.get_nearest_res_pnt(self.loc['x'], self.loc['y'])

        dx = np.sign(nearest_res[0] - self.loc['x'])
        dy = np.sign(nearest_res[1] - self.loc['y'])

        new_x = self.loc['x'] + dx
        new_y = self.loc['y'] + dy

        if abs(new_x - self.loc['x']) <= 1 and abs(new_y - self.loc['y']) <= 1:
            if (new_x, new_y) in self.grid.resources:
                self.grid.remove_resource(new_x, new_y)
                self.cumulative_reward += 10

        self.loc['x'] += dx
        self.loc['y'] += dy

        self.path.append((self.loc['x'], self.loc['y']))

    def get_movements(self):
        movements = []
        for i in range(1, len(self.path)):
            old_position = self.path[i - 1]
            new_position = self.path[i]
            movements.append((new_position))
        return movements

    def update(self):
        state = self.get_state()
        action = self.choose_action(state)
        old_distance = self.grid.get_distance_to_nearest_res_pnt(self.loc['x'], self.loc['y'])
        if action == 0:
            self.move_towards_resource()
        new_distance = self.grid.get_distance_to_nearest_res_pnt(self.loc['x'], self.loc['y'])
        if new_distance < old_distance:
            reward = 2
        elif new_distance == old_distance:
            reward = 0
        else:
            reward = -1
        reward *= 1 / (1 + self.grid.get_distance_to_nearest_res_pnt(self.loc['x'], self.loc['y']))
        self.cumulative_reward += reward
        self.past_rewards.append(reward)
        if len(self.past_rewards) > 100:
            self.past_rewards.pop(0)
        next_state = self.get_state()
        self.update_q_table(state, action, reward, next_state)
        self.exploration_decay = min(max(self.exploration_decay, 0.01), 0.99)
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)
        self.log(action, reward)
        self.history.append({
            'action': action,
            'reward': reward,
            'remaining_resources': self.grid.current_resource_count(),
            'position': (self.loc['x'], self.loc['y']),
            'cumulative_reward': self.cumulative_reward
        })

    def print_history(self):
        print(f"History for Agent {self.id}:")
        for i, (action, reward) in enumerate(self.history):
            print(f"Episode {i + 1}: Action = {action}, Reward = {reward}")


def run_simulation(num_runs, num_episodes, grid_width, grid_height, num_resources, num_agents):
    all_runs_cumulative_rewards = []
    all_agents_histories = []

    for run in range(num_runs):
        print(f"Starting Run {run + 1}/{num_runs}")
        simulation_grid = Grid(grid_width, grid_height, num_resources)

        agents = [SimpleAgent(id=i, grid=simulation_grid,
                              loc=(random.randint(0, grid_width - 1), random.randint(0, grid_height - 1))) for i in
                  range(num_agents)]

        run_cumulative_rewards = []
        run_histories = []

        for episode in range(num_episodes):
            episode_rewards = []
            episode_histories = []

            for agent in agents:
                agent.update()
                episode_rewards.append(agent.cumulative_reward)
                episode_histories.append({
                    'agent_id': agent.id,
                    'cumulative_reward': agent.cumulative_reward,
                    'history': agent.history[-1],
                })

            run_cumulative_rewards.append(np.mean(episode_rewards))
            run_histories.append(episode_histories)

        all_runs_cumulative_rewards.append(run_cumulative_rewards)
        all_agents_histories.append(run_histories)

    return all_runs_cumulative_rewards, all_agents_histories, simulation_grid, agents


# Parameters for the simulation
num_runs = 1
num_episodes = 100
grid_width = 10
grid_height = 10
num_resources = 3
num_agents = 2

# Run the simulation
all_runs_cumulative_rewards, all_agents_histories, simulation_grid, agents = run_simulation(num_runs, num_episodes,
                                                                                            grid_width, grid_height,
                                                                                            num_resources, num_agents)


def plot_agent_paths(agents, resources):
    plt.figure(figsize=(8, 8))

    # Plot resource points
    resource_x_coords = [res[0] for res in resources]
    resource_y_coords = [res[1] for res in resources]
    plt.scatter(resource_x_coords, resource_y_coords, color='red', marker='x', s=100, label='Resources')

    # Loop through each agent
    for agent in agents:
        # Get the agent's path
        path = agent.path

        # Plot the agent's path
        x_coords = [pos[0] for pos in path]
        y_coords = [pos[1] for pos in path]
        plt.plot(x_coords, y_coords, marker='o', linestyle='-', label=f'Agent {agent.id}')

    plt.title('Agent Movements and Resource Locations Over Episodes')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()


# Plotting Resource Changes
# def plot_resource_changes(agents_histories):
#     plt.figure(figsize=(10, 5))
#     episodes = list(range(len(agents_histories[0][0]['history'])))
#     for agent_hist in agents_histories[0]:
#         resources = [step['remaining_resources'] for step in agent_hist['history']]
#         plt.plot(episodes, resources, marker='o', label=f'Agent {agent_hist["agent_id"]}')
#     plt.title('Resource Changes Over Episodes')
#     plt.xlabel('Episode')
#     plt.ylabel('Remaining Resources')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# Assuming you have run the simulation and have the required data

resources = simulation_grid.get_initial_resource_positions()

plot_agent_paths(agents, resources)
# plot_resource_changes(all_agents_histories)

import random

# log_path = r"C:\Users\TR\Desktop\results"
log_path = r"C:\Users\tingram\Desktop\results"

inf_rate = 2

start_res = float(random.randint(10, 20))

start_ttd = float(random.randint(10, 20))

ttd_rate = .5

res_lose_rate = 1
hunger = res_lose_rate*2

max_res_gain = 5

size = 100
grid_size = (size, size)
w, h = size, size
vi = 0.025
vj = 0.025
z = .5


num_humans = 100

num_zombies = 5

epochs = 1

days = 365

loser_survival_rate = 0.25  # The loser keeps % of their resources

loser_death_rate = 0.5  # chance that the loser is killed

import os
import pandas as pd
from datetime import datetime
from abm import Simulation, ml, rl, gl
from config import log_path
from network_manager import NetworkManager
from mapping import SurfaceMapper


def main():
    # Initialize the simulation
    sim = Simulation()

    # Generate a unique folder name using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_log_path = os.path.join(log_path, timestamp)

    # Create the new folder
    os.makedirs(new_log_path, exist_ok=True)

    # Initialize an empty DataFrame for the encounter logs
    enc_df = pd.DataFrame(columns=["Epoch", "Day", "Entity 1", "Entity 2", "Interaction Type"])

    # Run the simulation and get the metrics dictionary and encounter logs
    metrics_list, encounter_logs = sim.run()

    # Convert the metrics dictionary to a DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # Gather network analysis statistics
    network_statistics = sim.network_manager.gather_statistics()

    # Convert the network statistics dictionary to a DataFrame
    network_statistics_df = pd.DataFrame(network_statistics)

    # Append the network statistics to the metrics DataFrame
    network_statistics_df.to_csv(os.path.join(new_log_path, "network_statistics.csv"), index=False)

    # Append the encounter logs for each epoch to the DataFrame
    for logs in encounter_logs:
        enc_df = enc_df._append(pd.DataFrame([str(record).split(',') for record in logs],
                                             columns=["Epoch", "Day", "Entity 1", "Entity 2", "Interaction Type"]))

    # Create dataframes for each log and write them to CSV files in the new folder
    move_df = pd.DataFrame([str(record).split(',') for record in ml.logs],
                           columns=["Epoch", "Day", "Entity", "Old X", "Old Y", "Old Z", "New X", "New Y", "New Z"])
    move_df.to_csv(os.path.join(new_log_path, "move_log.csv"), index=False)

    res_df = pd.DataFrame([str(record).split(',') for record in rl.logs],
                          columns=["Epoch", "Day", "Entity", "Resource Change", "Current Resources", "Reason"])
    res_df.to_csv(os.path.join(new_log_path, "res_log.csv"), index=False)

    grp_df = pd.DataFrame([str(record).split(',') for record in gl.logs],
                          columns=["Epoch", "Day", "Group", "Entity", "Action", "Reason"])
    grp_df.to_csv(os.path.join(new_log_path, "grp_log.csv"), index=False)

    # Write the metrics DataFrame to a CSV file
    metrics_df.to_csv(os.path.join(new_log_path, "metrics_log.csv"), index=False)

    # Write the encounter logs DataFrame to a CSV file
    enc_df.to_csv(os.path.join(new_log_path, "enc_log.csv"), index=False)

    # visualize the network
    sim.network_manager.visualize_network('human')
    sim.network_manager.visualize_network('zombie')

    # sim.network_manager.visualize_network('all')

    # general_graph_nodes = list(sim.network_manager.G.nodes())
    # human_graph_nodes = list(sim.network_manager.H.nodes())
    # zombie_graph_nodes = list(sim.network_manager.Z.nodes())
    #
    # print("Nodes in the general graph: ", general_graph_nodes)
    # print("Nodes in the human graph: ", human_graph_nodes)
    # print("Nodes in the zombie graph: ", zombie_graph_nodes)

    # general_graph_edges = len(list(sim.network_manager.G.edges()))
    human_graph_edges = len(list(sim.network_manager.H.edges()))
    zombie_graph_edges = len(list(sim.network_manager.Z.edges()))

    # print("Edges in the general graph: ", general_graph_edges)
    print("Edges in the human graph: ", human_graph_edges)
    print("Edges in the zombie graph: ", zombie_graph_edges)

    elev_data = sim.grid.surface
    mapper = SurfaceMapper(elev_data, new_log_path)
    mapper.plot_surface()

    return metrics_df, move_df, enc_df, res_df, grp_df


if __name__ == '__main__':
    metrics_df, move_df, enc_df, res_df, grp_df = main()

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
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class NetworkManager:
    def __init__(self):
        self.G = nx.Graph()  # Initialize an empty graph
        self.H = nx.Graph()  # Initialize an empty graph
        self.Z = nx.Graph()  # Initialize an empty graph
        print("Initialized an empty graph.")

    # Add an agent to the network
    def add_agent(self, agent_id, agent_type):
        self.G.add_node(agent_id, type=agent_type)
        if agent_type == 'human':
            self.H.add_node(agent_id, type=agent_type)
        elif agent_type == 'zombie':
            self.Z.add_node(agent_id, type=agent_type)

    def remove_agent(self, agent_id):
        if self.G.has_node(agent_id):
            self.G.nodes[agent_id]['status'] = 'dead'
            # print(f"Changed status of agent {agent_id} to 'dead'.")
        else:
            print(f"Node {agent_id} does not exist in the graph.")

    def add_interaction(self, agent1_id, agent2_id, interaction_type):
        self.G.add_edge(agent1_id, agent2_id, interaction=interaction_type)
        print(f"Added interaction between {agent1_id} and {agent2_id}.")

    def perform_analysis(self):
        # Calculate degree centrality
        degree_centrality = nx.degree_centrality(self.G)

        # Calculate betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(self.G)

        # Calculate closeness centrality
        closeness_centrality = nx.closeness_centrality(self.G)

        # Calculate eigenvector centrality
        eigenvector_centrality = nx.eigenvector_centrality(self.G)

        # Calculate clustering coefficient
        clustering_coefficient = nx.clustering(self.G)

        return {
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'closeness_centrality': closeness_centrality,
            'eigenvector_centrality': eigenvector_centrality,
            'clustering_coefficient': clustering_coefficient
        }

    def gather_statistics(self):
        analysis_results = self.perform_analysis()

        # for metric, values in analysis_results.items():
        #     print(f"{metric}:")
        #     for node, value in values.items():
        #         print(f"Node {node}: {value}")
        #     print("\n")

        return analysis_results

    def add_edge(self, a1, a2):
        if a1 in self.G and 'type' in self.G.nodes[a1] and a2 in self.G and 'type' in self.G.nodes[
            a2]:
            self.G.add_edge(a1, a2)
            agent1_type = self.G.nodes[a1]['type']
            agent2_type = self.G.nodes[a2]['type']
            if agent1_type == 'human' and agent2_type == 'human':
                self.H.add_edge(a1, a2)
            elif agent1_type == 'zombie' and agent2_type == 'zombie':
                self.Z.add_edge(a1, a2)
        else:
            print(f"Nodes {a1} and/or {a2} do not exist in the graph or do not have a 'type' attribute.")

    def remove_edge(self, node1, node2):
        self.G.remove_edge(node1, node2)
        print(f"Removed edge between {node1} and {node2}.")


    def visualize_network(self, network_type):
        # Select the appropriate graph
        if network_type == 'human':
            G = self.H
        elif network_type == 'zombie':
            G = self.Z
        else:
            G = self.G

        # Generate random 3D positions for each node
        pos = {node: (np.random.rand(), np.random.rand(), np.random.rand()) for node in G.nodes()}

        # Set up 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Extract node positions
        xs, ys, zs = [], [], []
        for node, (x, y, z) in pos.items():
            xs.append(x)
            ys.append(y)
            zs.append(z)

        # Plot nodes
        ax.scatter(xs, ys, zs)

        # Plot edges
        for edge in G.edges():
            x, y, z = zip(*[pos[node] for node in edge])
            ax.plot(x, y, z, color='black')  # Customize color as needed

        # Customize the axes and display
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()



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

vi = vi
vj = vj
z = z
w = w
h = h

#generate a 2d grid of perlin noise that is 20 by 20
def generate_noise(w, h, vi, vj, z):
    noise = p.Perlin(1414)
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


import datetime
import random
from config import log_path
import csv
import os


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

import os

# Specify the directory you want to combine scripts from
directory = r'C:\Users\tingram\Desktop\Captains Log\UWYO\GIT\modeling-v2'

# Specify the output file
output_file = 'combined_script.py'

# Get a list of all Python files in the directory
python_files = [f for f in os.listdir(directory) if f.endswith('.py')]

# Open the output file in write mode
with open(output_file, 'w') as outfile:
    for fname in python_files:
        # Open each Python file in read mode
        with open(os.path.join(directory, fname)) as infile:
            # Write the contents of the Python file to the output file
            outfile.write(infile.read())
            # Write a newline character to separate scripts
            outfile.write('\n')

print(f"Combined scripts saved to {output_file}")
import random
import networkx as nx
from agent_testing import Grid, SimpleAgent, run_simulation
from abm import Simulation, Human, Zombie, NetworkManager, interact, update_status

from config import grid_size, start_res, start_ttd, max_res_gain, ttd_rate, res_lose_rate, inf_rate, w, h, \
    vi, vj, z, num_humans, num_zombies, epochs, days, hunger
from surface_noise import generate_noise

# Enhanced Grid class with noise and resource management
class EnhancedGrid(Grid):
    def __init__(self, width, height, num_resources):
        super().__init__(width, height, num_resources)
        self.surface = generate_noise(width, height, vi, vj, z)

    def get_elevation(self, x, y):
        return self.surface[x][y]

# Enhanced Simulation class integrating logic from SimpleAgent
class EnhancedSimulation(Simulation):
    def __init__(self, num_humans, num_zombies, grid_size):
        super().__init__(num_humans, num_zombies)
        self.grid = EnhancedGrid(grid_size[0], grid_size[1], 50)  # Set resource count example
        self.agents = [
            SimpleAgent(i, self.grid, (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))) for i
            in range(num_humans + num_zombies)]

    def run_day(self):
        # Implement day activities combining logic from both systems
        for agent in self.agents:
            agent.update()  # SimpleAgent logic for movement and resource collection
            self.interact_agents(agent)  # Interaction logic potentially using network relationships

    def interact_agents(self, agent):
        # Enhanced interaction logic considering proximity and network
        for other_agent in self.agents:
            if agent != other_agent and agent.grid.is_adjacent(agent, other_agent):
                interact(self, agent, other_agent)  # Utilize the interaction logic defined in abm.py

# Main function to run the enhanced simulation
def main_simulation():
    # num_humans = 10
    # num_zombies = 5
    # grid_size = (20, 20)  # Example grid size
    simulation = EnhancedSimulation(num_humans, num_zombies, grid_size)
    for _ in range(100):  # Run for 100 days
        simulation.run_day()

if __name__ == "__main__":
    main_simulation()

