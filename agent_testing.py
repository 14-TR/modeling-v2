import numpy as np
import random

from matplotlib import pyplot as plt


class Grid:
    def __init__(self, width, height, num_resources):
        self.width = width
        self.height = height
        self.resources = self.generate_resources(num_resources)

    def plot_resources(self):
        plt.figure(figsize=(6, 6))
        for res in self.resources:
            plt.scatter(res[0], res[1], color='red')
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.grid(True)
        plt.title('Resources on Grid')
        plt.show()
    def generate_resources(self, num_resources):
        return {(random.randint(0, self.width - 1), random.randint(0, self.height - 1)) for _ in range(num_resources)}

    def remove_resource(self, x, y):
        if (x, y) in self.resources:
            self.resources.remove((x, y))

    def get_nearest_res_pnt(self, x, y):
        nearest_resource = None
        min_distance = float('inf')
        for res in self.resources:
            distance = np.sqrt((res[0] - x)**2 + (res[1] - y)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_resource = res
        return nearest_resource if nearest_resource else (0, 0)

    def get_distance_to_nearest_res_pnt(self, x, y):
        nearest_resource = self.get_nearest_res_pnt(x, y)
        return np.sqrt((nearest_resource[0] - x)**2 + (nearest_resource[1] - y)**2)


class SimpleAgent:
    def __init__(self, id, grid, loc=(0, 0), learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        self.id = id
        self.grid = grid
        self.loc = {'x': loc[0], 'y': loc[1]}  # Agent's location on the grid
        self.q_table = np.zeros((grid.width * grid.height, 2))  # Initialize Q-table
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.cumulative_reward = 0
        self.past_rewards = []
        self.history = []

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
            return random.choice([0, 1])  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        future_rewards = np.max(self.q_table[next_state])
        estimated_q = self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * (reward + self.discount_factor * future_rewards - estimated_q)

    def move_towards_resource(self):
        nearest_res = self.grid.get_nearest_res_pnt(self.loc['x'], self.loc['y'])
        dx = np.sign(nearest_res[0] - self.loc['x'])
        dy = np.sign(nearest_res[1] - self.loc['y'])
        new_x = self.loc['x'] + dx
        new_y = self.loc['y'] + dy

        if (new_x, new_y) in self.grid.resources:
            self.grid.remove_resource(new_x, new_y)
            self.cumulative_reward += 10

        self.loc['x'] += dx
        self.loc['y'] += dy

    def update(self):
        state = self.get_state()
        action = self.choose_action(state)
        old_distance = self.grid.get_distance_to_nearest_res_pnt(self.loc['x'], self.loc['y'])

        if action == 0:
            self.move_towards_resource()

        new_distance = self.grid.get_distance_to_nearest_res_pnt(self.loc['x'], self.loc['y'])

        # Adjust the reward based on the action and the resulting distance
        if new_distance < old_distance:
            reward = 2  # Larger reward for moving towards a resource
        elif new_distance == old_distance:
            reward = .01 # Smaller reward for staying in the same place
        else:
            reward = -1  # Penalty for moving away from a resource

        # Apply a time decay to the reward
        reward *= 1 / (1 + self.grid.get_distance_to_nearest_res_pnt(self.loc['x'], self.loc['y']))

        self.cumulative_reward += reward
        self.past_rewards.append(reward)
        if len(self.past_rewards) > 100:  # Keep the size of past_rewards to the last 100 rewards
            self.past_rewards.pop(0)

        next_state = self.get_state()
        self.update_q_table(state, action, reward, next_state)

        # Ensure the exploration decay rate is within the range [0.01, 0.99]
        self.exploration_decay = min(max(self.exploration_decay, 0.01), 0.99)

        # Update the exploration rate
        self.exploration_rate *= self.exploration_decay
        # Ensure the exploration rate never falls below the minimum exploration rate
        self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)

        self.log(action, reward)

    def print_history(self):
        print(f"History for Agent {self.id}:")
        for i, (action, reward) in enumerate(self.history):
            print(f"Episode {i+1}: Action = {action}, Reward = {reward}")


def run_simulation(num_runs, num_episodes, grid_width, grid_height, num_resources, num_agents):
    all_runs_cumulative_rewards = []
    all_agents = []

    for run in range(num_runs):
        print(f"Starting Run {run+1}/{num_runs}")

        simulation_grid = Grid(grid_width, grid_height, num_resources)
        agents = [SimpleAgent(id=i, grid=simulation_grid, loc=(random.randint(0, grid_width - 1), random.randint(0, grid_height - 1))) for i in range(num_agents)]

        run_cumulative_rewards = []  # Initialize run_cumulative_rewards here

        # Load the Q-table from the previous run for each agent
        if run > 0:
            for agent in agents:
                agent.load_q_table(f'q_table_agent_{agent.id}_run_{run}.npy')

        for episode in range(num_episodes):
            episode_rewards = []

            for agent in agents:
                agent.update()
                episode_rewards.append(agent.cumulative_reward)

            run_cumulative_rewards.append(np.mean(episode_rewards))

        # Save the Q-table at the end of the run for each agent
        for agent in agents:
            agent.save_q_table(f'q_table_agent_{agent.id}_run_{run+1}.npy')

        all_runs_cumulative_rewards.append(run_cumulative_rewards)
        all_agents.append(agents)

    return all_runs_cumulative_rewards, all_agents

# Parameters for the simulation
num_runs = 1
num_episodes = 1000
grid_width = 10
grid_height = 10
num_resources = 5
num_agents = 2

# Run the simulation
cumulative_rewards, agents = run_simulation(num_runs, num_episodes, grid_width, grid_height, num_resources, num_agents)

# Example: Analyze the cumulative rewards
# Plotting the average cumulative reward per episode over all runs
average_cumulative_rewards = np.mean(cumulative_rewards, axis=0)

# Print summary statistics
print("Simulation Summary:")
print(f"Total Runs: {num_runs}")
print(f"Episodes per Run: {num_episodes}")

# Average Cumulative Reward Analysis
print("\nAverage Cumulative Reward Analysis:")
for episode in range(0, num_episodes, 10):  # Print the average cumulative reward every 10 episodes
    print(f"Episode {episode + 1}: Average Cumulative Reward: {average_cumulative_rewards[episode]:.2f}")

# Plotting for visualization (if you're able to generate plots)
plt.plot(average_cumulative_rewards)
plt.xlabel('Episode')
plt.ylabel('Average Cumulative Reward')
plt.title('Average Cumulative Reward Over Episodes Across Runs')
plt.grid(True)
plt.show()

agents[0][0].plot_history()
agents[0][0].print_history()

simulation_grid = Grid(grid_width, grid_height, num_resources)
simulation_grid.plot_resources()
