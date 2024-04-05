import pandas as pd

from env import Grid, DayTracker, Epoch
from ents import Human, Zombie, Group, entities
from config import grid_size, num_humans, num_zombies, epochs, days
from events import interact
from log import el


def reset_entities():
    entities.clear()


class Simulation:

    def __init__(self, humans=num_humans, zombies=num_zombies, e=epochs, d=days):
        self.grid = Grid(grid_size=grid_size)
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
        for zombie in self.zombies:
            self.grid.add_ent(zombie)

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

        DayTracker.increment_day()

        # Interaction between humans and zombies
        for human in list(self.humans):  # Create a copy of the list
            for zombie in self.zombies:
                interact(self, human, zombie)

        print(DayTracker.get_current_day(),len(self.humans), len(self.zombies))


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

    def run(self):
        # Initialize the list for metrics
        metrics_list = []

        for epoch in range(self.epochs):
            metrics = {}

            peak_zombies = 0
            peak_groups = 0
            for _ in range(self.days):
                self.simulate_day()
                #

                peak_zombies = max(peak_zombies, len(self.zombies))
                peak_groups = max(peak_groups, len(Group.groups))

            # Get the final counts of humans and zombies
            ending_num_humans = len(self.humans)
            ending_num_zombies = len(self.zombies)

            # Calculate encounter types from entity attributes
            enc_types = {'love': 0, 'war': 0, 'rob': 0, 'esc': 0, 'kill': 0, 'infect': 0}
            for log in el.logs:
                enc_types[log.action] += 1

            # Log the metrics for this epoch
            metrics['Epoch'] = epoch
            metrics['End_Day'] = DayTracker.get_current_day()
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

            if metrics['Ending_Num_Humans'] == 0:
                print(len(self.humans), len(self.zombies))
                reset_entities()
                break

        # Return the list of metrics dictionaries
        return metrics_list
