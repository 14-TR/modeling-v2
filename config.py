import datetime
import random
import os

# log_path = r"C:\Users\TR\Desktop\results"
log_path = r"C:\Users\tingram\Desktop\results"
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
new_log_path = os.path.join(log_path, timestamp)

inf_rate = 2

def get_random_res():
    return random.randint(10, 20)

start_res = get_random_res()

start_ttd = random.randint(10, 20)

ttd_rate = .005

res_lose_rate = .005

max_res_gain = 5

size = 100
grid_size = (size, size)
w, h = size, size
vi = 0.025
vj = 0.025
z = 5

num_humans = 100

num_zombies = 1

epochs = 384

days = 100

loser_survival_rate = 0.25  # The loser keeps % of their resources

loser_death_rate = 0.5  # chance that the loser is killed

# Define markers and colors for each encounter type
markers = {
    'Type1': 'o',  # Circle
    'Type2': 's',  # Square
    'Type3': '^',  # Triangle
    'Type4': 'd',  # Diamond
}

colors = {
    'Type1': 'blue',
    'Type2': 'green',
    'Type3': 'orange',
    'Type4': 'purple',
}

# Default values if a type is not in the dictionary
default_marker = '^'
default_color = (0.7, 0.5, 0.8, 0.95)
