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
            human.move(self)
            human.update_status(self)
        for zombie in list(self.zombies):
            zombie.move()
            zombie.update_status(self)

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

    def handle_turn_into_zombie(self, human, new_zombie):
        # Handle the human to zombie transition within the simulation
        self.humans.remove(human)
        self.zombies.append(new_zombie)
        # Any other updates related to this transition should be handled here

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

