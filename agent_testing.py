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
