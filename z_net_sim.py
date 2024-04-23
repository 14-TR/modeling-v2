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
