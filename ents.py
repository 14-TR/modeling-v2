import random

from env import Grid
from util import IDGenerator


class Entity:

    id_gen = IDGenerator()

    def __init__(self, id_gen):
        self.id = id_gen.generate_id()
        self.loc = {'x': 0, 'y': 0, 'z': 0}  # Default location
        self.att = {'res': 0, 'ttd': 0}  # Default attributes
        self.is_z = False  # Default zombie flag
        self.enc = {'luv': 0, 'war': 0, 'rob': 0, 'esc': 0, 'win': 0}  # Default encounters
        self.xp = {'luv_xp': 0 , 'war_xp': 0, 'esc_xp': 0, 'win_xp': 0}
        self.node = {'human': False, 'zombie': False}  # Default node flags
        self.grp = {}  # Default groups
        self.net = {} # Default network connections
        self.res = 0
        # self.grid = Grid()

    def move(self):
        if self.is_z:
            Zombie.move()
        else:
            Human.move()


class Human(Entity):

    def __init__(self):
        super().__init__()

    def move(self):
        if self.grp and self.prob_move_grp():
            self.move_grp()
        elif self.prob_move_res():
            self.move_res()
        else:
            self.move_rand()

        self.res -= .5

    def prob_move_res(self):
        return random.random() < (1 - self.res / 10)

    def prob_move_grp(self):
        collective_res = sum(member.res for member in self.grp.values()) / len(self.grp)
        return collective_res < 5

    def move_rand(self):
        # Random movement logic, modifying x and y coordinates
        step_range = (-4, -3, -2, -1, 1, 2, 3, 4)  # Define the range of steps (inclusive). Adjust as needed.
        dx = random.randint(*step_range)  # Step size for x direction
        dy = random.randint(*step_range)  # Step size for y direction

        # Update location
        self.loc[ 'x' ] += dx
        self.loc[ 'y' ] += dy

    def move_res(self):
        # Assuming the Grid class has a method get_nearest_resource_point() that returns the closest resource point's (x, y) location
        nearest_res_x, nearest_res_y = self.grid.get_nearest_res_pnt(self.loc[ 'x' ], self.loc[ 'y' ])

        # Calculate direction to move towards the resource point. This is a simple version that moves one step towards the resource point.
        dx = 1 if nearest_res_x > self.loc[ 'x' ] else -1 if nearest_res_x < self.loc[ 'x' ] else 0
        dy = 1 if nearest_res_y > self.loc[ 'y' ] else -1 if nearest_res_y < self.loc[ 'y' ] else 0

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

        # Update location towards the group's average position
        self.loc[ 'x' ] += dx
        self.loc[ 'y' ] += dy

        print(f"Moving with group: New location ({self.loc[ 'x' ]}, {self.loc[ 'y' ]})")


class Zombie(Entity):

    def __init__(self):
        super().__init__()

    def move(self):
        if self.prob_move_prey():
            self.move_prey()
        else:
            self.move_rand()

        self.res -= 1.5

    def prob_move_grp(self):
        collective_res = sum(member.res for member in self.grp.values()) / len(self.grp)
        return collective_res < 5

    def prob_move_prey(self):
        return random.random() < (1 - self.res / 10)

    def move_rand(self):
        # Random movement logic, modifying x and y coordinates
        step_range = (-2, -1, 1, 2)  # Define the range of steps (inclusive). Adjust as needed.
        dx = random.randint(*step_range)  # Step size for x direction
        dy = random.randint(*step_range)  # Step size for y direction

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
        # Assuming the Grid class has a method get_nearest_prey() that returns the closest human's (x, y) location
        nearest_prey_x, nearest_prey_y = self.grid.get_nearest_prey(self.loc[ 'x' ], self.loc[ 'y' ])

        # Calculate direction to move towards the human. This is a simple version that moves one step towards the human.
        dx = 1 if nearest_prey_x > self.loc[ 'x' ] else -1 if nearest_prey_x < self.loc[ 'x' ] else 0
        dy = 1 if nearest_prey_y > self.loc[ 'y' ] else -1 if nearest_prey_y < self.loc[ 'y' ] else 0

        # Update location towards the human
        self.loc[ 'x' ] += dx
        self.loc[ 'y' ] += dy

        print(f"Moving towards prey: New location ({self.loc[ 'x' ]}, {self.loc[ 'y' ]})")






