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

    def turn_into_zombie(self):
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
