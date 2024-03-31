from env import Grid
from ents import Human, Zombie, Group
from config import grid_size, num_humans, num_zombies, epochs, days
from events import interact


class Simulation:
    def __init__(self, humans=num_humans, zombies=num_zombies, e=epochs, d=days):
        self.grid = Grid(*grid_size)
        self.humans = [Human() for _ in range(humans)]
        self.zombies = [Zombie() for _ in range(zombies)]
        self.epochs = e
        self.days = d

    def simulate_day(self):
        for human in self.humans:
            human.move()
            human.update_status()
        for zombie in self.zombies:
            zombie.move()
            zombie.update_status()

        # Interaction between humans and zombies
        for human in self.humans:
            for zombie in self.zombies:
                interact(human, zombie)

        # Interaction between humans
        for i in range(len(self.humans)):
            for j in range(i+1, len(self.humans)):
                interact(self.humans[i], self.humans[j])

        # Interaction between zombies
        for i in range(len(self.zombies)):
            for j in range(i+1, len(self.zombies)):
                interact(self.zombies[i], self.zombies[j])

    def run(self):
        for _ in range(self.days):
            for _ in range(self.epochs):
                self.simulate_day()