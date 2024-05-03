# main.py

import config
from agent_testing import run_simulation, plot_agent_paths

def main():
    # Simulation parameters from config
    grid_width = config.grid_size[0]
    grid_height = config.grid_size[1]
    num_resources = config.num_resources
    num_agents = config.num_humans + config.num_zombies

    # Run the simulation
    cumulative_rewards, agents_histories, simulation_grid, agents = run_simulation(
        num_runs=1,
        num_episodes=100,
        grid_width=grid_width,
        grid_height=grid_height,
        num_resources=num_resources,
        num_agents=num_agents
    )

    # Plotting agent paths to visualize their movements
    resources = simulation_grid.get_initial_resource_positions()
    plot_agent_paths(agents, resources)

if __name__ == "__main__":
    main()
