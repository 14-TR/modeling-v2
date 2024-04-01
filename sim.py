import pandas as pd

from env import Grid, DayTracker
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

    def reset_entitiies(self):
        entities.clear()

    def run(self):
        # Initialize the DataFrame for metrics
        metrics = pd.DataFrame(columns=['Epoch', 'Total_Days', 'Ending_Num_Humans', 'Ending_Num_Zombies', 'Peak_Zombies', 'Peak_Groups', 'Love', 'War', 'Rob', 'Esc', 'Kill', 'Infect'])

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

            DayTracker.increment_day()

            # Get the final counts of humans and zombies
            ending_num_humans = len(self.humans)
            ending_num_zombies = len(self.zombies)

            # Calculate encounter types from entity attributes
            enc_types = {'love': 0, 'war': 0, 'rob': 0, 'esc': 0, 'kill': 0, 'infect': 0}
            for log in el.logs:
                enc_types[log.action] += 1

            # Log the metrics for this epoch
            metrics = metrics.append({'Epoch': epoch, 'Total_Days': self.days, 'Ending_Num_Humans': ending_num_humans, 'Ending_Num_Zombies': ending_num_zombies, 'Peak_Zombies': peak_zombies, 'Peak_Groups': peak_groups, 'Love': enc_types['love'], 'War': enc_types['war'], 'Rob': enc_types['rob'], 'Esc': enc_types['esc'], 'Kill': enc_types['kill'], 'Infect': enc_types['infect']}, ignore_index=True)

