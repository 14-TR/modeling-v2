import random

log_path = "logs/"

inf_rate = 0.3

start_res = random.randint(10, 20)

start_ttd = random.randint(10, 20)

max_res_gain = 5

size = 50
grid_size = (size, size)
vi = 0.025
vj = 0.025
z = 4

num_humans = 10

num_zombies = 5

epochs = 1

days = 365

loser_survival_rate = 0.25  # The loser keeps % of their resources

loser_death_rate = 0.5  # chance that the loser is killed